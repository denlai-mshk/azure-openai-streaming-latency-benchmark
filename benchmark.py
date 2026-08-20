#!/usr/bin/env python3
"""Azure OpenAI response-ready benchmark.

Sends structured Chat Completions requests to Azure OpenAI deployments using the
current Azure CLI login, measures complete-response readiness, validates assistant
output against template schemas, and renders an HTML report. Client-observed latency
includes network and Azure service time; authentication is outside the request timer.

Usage:
    python benchmark.py --config config.yaml --out .
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import httpx
import yaml
from azure.core.exceptions import AzureError
from azure.identity import AzureCliCredential
from jsonschema import Draft202012Validator, ValidationError

AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


# ---------------------------------------------------------------------------
# Config & template loading
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    deployment: str
    display_name: str
    reasoning_effort: str | None = None
    supports_reasoning_effort: bool = False


@dataclass
class BenchmarkConfig:
    endpoint: str
    api_version: str
    tenant_id: str | None
    token_budget: int
    request_timeout_s: float
    models: list[ModelConfig]
    templates: list[str]
    iterations: int = 10


@dataclass
class PromptTemplate:
    name: str
    display_name: str
    system: str
    queries: list[str]
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    validation: dict[str, Any] | None = None


@dataclass
class IterationResult:
    schema_version: int
    template: str
    model: str
    display_name: str
    iteration: int
    warmup: bool
    query: str
    response_mode: str
    success: bool
    transport_success: bool
    content_received: bool
    json_parse_valid: bool | None
    schema_valid: bool | None
    response_ready: bool
    http_response_complete_ms: float | None
    json_parse_ms: float | None
    schema_validation_ms: float | None
    response_ready_ms: float | None
    first_token_ms: float | None
    last_content_ms: float | None
    stream_complete_ms: float | None
    last_token_ms: float | None
    http_status: int | None
    failure_phase: str | None
    error: str | None
    output_preview: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None


@dataclass
class ValidationResult:
    json_parse_valid: bool | None
    schema_valid: bool | None
    json_parse_ms: float
    schema_validation_ms: float
    error: str | None


@dataclass
class RequestMeasurement:
    response_mode: str
    success: bool
    transport_success: bool
    content_received: bool
    json_parse_valid: bool | None
    schema_valid: bool | None
    response_ready: bool
    http_response_complete_ms: float | None
    json_parse_ms: float | None
    schema_validation_ms: float | None
    response_ready_ms: float | None
    first_token_ms: float | None
    last_content_ms: float | None
    stream_complete_ms: float | None
    http_status: int | None
    failure_phase: str | None
    error: str | None
    output_preview: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path) -> BenchmarkConfig:
    raw = _load_yaml(path)
    iterations = raw.get("iterations", 10)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be an integer greater than or equal to 1")
    models = [
        ModelConfig(
            deployment=str(model["deployment"]),
            display_name=str(model["display_name"]),
            reasoning_effort=(
                str(model["reasoning_effort"])
                if model.get("reasoning_effort") is not None
                else None
            ),
            supports_reasoning_effort=bool(
                model.get(
                    "supports_reasoning_effort",
                    model.get("reasoning_effort") is not None,
                )
            ),
        )
        for model in raw["models"]
    ]
    # Accept legacy `max_output_tokens` key for backward compatibility.
    budget = raw.get("token_budget", raw.get("max_output_tokens", 2000))
    return BenchmarkConfig(
        endpoint=str(raw["endpoint"]).rstrip("/"),
        api_version=str(raw["api_version"]),
        tenant_id=str(raw["tenant_id"]) if raw.get("tenant_id") else None,
        token_budget=int(budget),
        request_timeout_s=float(raw.get("request_timeout_s", 180)),
        models=models,
        templates=list(raw["templates"]),
        iterations=iterations,
    )


def load_template(path: Path) -> PromptTemplate:
    raw = _load_yaml(path)
    queries = [str(query) for query in raw["queries"]]
    if not queries:
        raise ValueError(f"Template must contain at least one query: {path}")
    validation = raw.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError(f"Template validation must be a mapping: {path}")
        schema = validation.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"Template validation.schema must be a mapping: {path}")
        Draft202012Validator.check_schema(schema)
    return PromptTemplate(
        name=str(raw["name"]),
        display_name=str(raw.get("display_name", raw["name"])),
        system=str(raw["system"]),
        queries=queries,
        reasoning_effort=(
            str(raw["reasoning_effort"])
            if raw.get("reasoning_effort") is not None
            else None
        ),
        temperature=(
            float(raw["temperature"])
            if raw.get("temperature") is not None
            else None
        ),
        top_p=float(raw["top_p"]) if raw.get("top_p") is not None else None,
        validation=validation,
    )


def iter_benchmark_queries(
    queries: list[str],
    iterations: int,
) -> Iterator[tuple[int, str]]:
    for iteration in range(1, iterations + 1):
        yield iteration, queries[(iteration - 1) % len(queries)]


def _token_tenant_id(token: str) -> str:
    """Read the tenant claim from an Entra JWT without logging the token."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        tenant_id = claims["tid"]
    except (
        binascii.Error,
        IndexError,
        KeyError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError("Azure CLI returned a token without a readable tenant claim") from exc
    if not isinstance(tenant_id, str) or not tenant_id:
        raise RuntimeError("Azure CLI returned a token without a readable tenant claim")
    return tenant_id


def acquire_azure_cli_token(tenant_id: str | None) -> tuple[str, str]:
    """Acquire one Azure OpenAI token from the current Azure CLI login."""
    credential = (
        AzureCliCredential(tenant_id=tenant_id)
        if tenant_id
        else AzureCliCredential()
    )
    try:
        token = credential.get_token(AZURE_OPENAI_SCOPE).token
    except AzureError as exc:
        raise RuntimeError(
            "Azure CLI authentication failed. Run "
            "'az login --tenant <tenant-id> --use-device-code' and try again."
        ) from exc

    token_tenant_id = _token_tenant_id(token)
    if tenant_id and token_tenant_id.casefold() != tenant_id.casefold():
        raise RuntimeError(
            f"Azure CLI token tenant {token_tenant_id} does not match configured "
            f"tenant_id {tenant_id}"
        )
    return token, token_tenant_id


# ---------------------------------------------------------------------------
# Request building & streaming measurement
# ---------------------------------------------------------------------------


def build_url(cfg: BenchmarkConfig, deployment: str) -> str:
    return (
        f"{cfg.endpoint}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={cfg.api_version}"
    )


def build_payload(
    model: ModelConfig,
    template: PromptTemplate,
    query: str,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": template.system},
            {"role": "user", "content": query},
        ],
        "stream": True,
    }
    # Reasoning-capable models require max_completion_tokens and may reject sampling controls.
    if model.supports_reasoning_effort:
        payload["max_completion_tokens"] = cfg.token_budget
        reasoning_effort = template.reasoning_effort or model.reasoning_effort
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["max_tokens"] = cfg.token_budget
        if template.temperature is not None:
            payload["temperature"] = template.temperature
        if template.top_p is not None:
            payload["top_p"] = template.top_p
    return payload


def parse_and_validate_output(
    content: str,
    template: PromptTemplate,
    clock: Any = time.perf_counter,
) -> ValidationResult:
    if template.validation is None:
        return ValidationResult(None, None, 0.0, 0.0, None)

    parse_start = clock()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        parse_ms = (clock() - parse_start) * 1000
        return ValidationResult(
            False,
            None,
            parse_ms,
            0.0,
            f"invalid assistant JSON: {exc.msg}",
        )
    parse_ms = (clock() - parse_start) * 1000

    validation_start = clock()
    try:
        Draft202012Validator(template.validation["schema"]).validate(parsed)
    except ValidationError as exc:
        validation_ms = (clock() - validation_start) * 1000
        location = ".".join(str(part) for part in exc.absolute_path) or "$"
        return ValidationResult(
            True,
            False,
            parse_ms,
            validation_ms,
            f"schema validation failed at {location}: rule {exc.validator}",
        )
    validation_ms = (clock() - validation_start) * 1000
    return ValidationResult(True, True, parse_ms, validation_ms, None)


def measure_streaming(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    template: PromptTemplate,
    clock: Any = time.perf_counter,
) -> RequestMeasurement:
    first_token_ms: float | None = None
    last_content_ms: float | None = None
    chunks: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    done_received = False
    t0 = clock()
    try:
        with client.stream(
            "POST", url, headers=headers, json=payload, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")[:400]
                return RequestMeasurement(
                    response_mode="streaming", success=False,
                    transport_success=True, content_received=False,
                    json_parse_valid=False, schema_valid=None,
                    response_ready=False, http_response_complete_ms=None,
                    json_parse_ms=None, schema_validation_ms=None,
                    response_ready_ms=None, first_token_ms=None,
                    last_content_ms=None, stream_complete_ms=None,
                    http_status=response.status_code, failure_phase="http",
                    error=body, output_preview=None, finish_reason=None, usage=None,
                )
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_received = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    return RequestMeasurement(
                        response_mode="streaming", success=False,
                        transport_success=True, content_received=bool(chunks),
                        json_parse_valid=False, schema_valid=None,
                        response_ready=False, http_response_complete_ms=None,
                        json_parse_ms=None, schema_validation_ms=None,
                        response_ready_ms=None, first_token_ms=first_token_ms,
                        last_content_ms=last_content_ms, stream_complete_ms=None,
                        http_status=response.status_code, failure_phase="stream_event",
                        error=f"invalid SSE JSON: {exc.msg}",
                        output_preview="".join(chunks)[:120] or None,
                        finish_reason=finish_reason, usage=usage,
                    )
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                content = (choice.get("delta") or {}).get("content")
                if isinstance(content, str) and content:
                    elapsed_ms = (clock() - t0) * 1000
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms
                    last_content_ms = elapsed_ms
                    chunks.append(content)
            stream_complete_ms = (clock() - t0) * 1000
            status = response.status_code
    except httpx.HTTPError as exc:
        return RequestMeasurement(
            response_mode="streaming", success=False,
            transport_success=False, content_received=bool(chunks),
            json_parse_valid=False, schema_valid=None, response_ready=False,
            http_response_complete_ms=None, json_parse_ms=None,
            schema_validation_ms=None, response_ready_ms=None,
            first_token_ms=first_token_ms, last_content_ms=last_content_ms,
            stream_complete_ms=None, http_status=None, failure_phase="transport",
            error=f"{type(exc).__name__}: {exc}",
            output_preview="".join(chunks)[:120] or None,
            finish_reason=finish_reason, usage=usage,
        )

    content = "".join(chunks)
    if not done_received:
        return RequestMeasurement(
            "streaming", False, True, bool(content), False, None, False,
            None, None, None, None, first_token_ms,
            last_content_ms, stream_complete_ms, status, "stream_terminal",
            "stream closed before [DONE]", content[:120] or None,
            finish_reason, usage,
        )
    if not content:
        return RequestMeasurement(
            "streaming", False, True, False, False, None, False,
            None, None, None, None, None, None,
            stream_complete_ms, status, "content", "no assistant content received",
            None, finish_reason, usage,
        )

    validation = parse_and_validate_output(content, template, clock)
    response_ready = (
        validation.json_parse_valid is not False
        and validation.schema_valid is not False
    )
    ready_ms = (clock() - t0) * 1000 if response_ready else None
    failure_phase = None
    if validation.json_parse_valid is False:
        failure_phase = "json_parse"
    elif validation.schema_valid is False:
        failure_phase = "schema"
    return RequestMeasurement(
        "streaming", response_ready, True, True,
        validation.json_parse_valid, validation.schema_valid, response_ready,
        None, validation.json_parse_ms,
        validation.schema_validation_ms, ready_ms, first_token_ms,
        last_content_ms, stream_complete_ms, status, failure_phase,
        validation.error, content[:120], finish_reason, usage,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summarize(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "min": min(vals),
        "max": max(vals),
    }


def build_model_summary(
    model: ModelConfig,
    results: list[IterationResult],
) -> dict[str, Any]:
    ready = [result for result in results if result.response_ready]
    http_values = [
        result.http_response_complete_ms
        for result in results
        if result.http_response_complete_ms is not None
    ]
    ready_values = [
        result.response_ready_ms
        for result in ready
        if result.response_ready_ms is not None
    ]
    validation_values = [
        (result.json_parse_ms or 0.0) + (result.schema_validation_ms or 0.0)
        for result in results
        if result.content_received
    ]
    first_values = [
        result.first_token_ms
        for result in results
        if result.first_token_ms is not None
    ]
    stream_values = [
        result.stream_complete_ms
        for result in results
        if result.stream_complete_ms is not None
    ]
    last_content_values = [
        result.last_content_ms
        for result in results
        if result.last_content_ms is not None
    ]
    post_first_values = [
        result.last_content_ms - result.first_token_ms
        for result in results
        if result.last_content_ms is not None and result.first_token_ms is not None
    ]
    thinking_estimate_values = [
        result.first_token_ms
        for result in ready
        if result.first_token_ms is not None
    ]
    composing_estimate_values = [
        result.last_content_ms - result.first_token_ms
        for result in ready
        if result.last_content_ms is not None and result.first_token_ms is not None
    ]
    completion_tail_values = [
        result.stream_complete_ms - result.last_content_ms
        for result in ready
        if result.stream_complete_ms is not None and result.last_content_ms is not None
    ]
    response_completion_wait_values = [
        result.stream_complete_ms
        for result in ready
        if result.stream_complete_ms is not None
    ]
    usable_ttlt_values = [
        result.last_content_ms
        for result in ready
        if result.last_content_ms is not None
    ]
    failure_counts: dict[str, int] = {}
    failure_examples: dict[str, str] = {}
    for result in results:
        if result.failure_phase:
            failure_counts[result.failure_phase] = failure_counts.get(result.failure_phase, 0) + 1
            if result.error and result.failure_phase not in failure_examples:
                failure_examples[result.failure_phase] = result.error[:240]
    completion_tokens = [
        float(result.usage["completion_tokens"])
        for result in results
        if result.usage
        and isinstance(result.usage.get("completion_tokens"), (int, float))
    ]
    total = len(results)
    ready_count = len(ready)
    return {
        "deployment": model.deployment,
        "display_name": model.display_name,
        "success": ready_count,
        "total": total,
        "response_ready": ready_count,
        "response_ready_rate": ready_count / total if total else 0.0,
        "transport_success": sum(result.transport_success for result in results),
        "content_received": sum(result.content_received for result in results),
        "json_parse_valid": sum(result.json_parse_valid is True for result in results),
        "schema_valid": sum(result.schema_valid is True for result in results),
        "http_response_complete_ms": summarize(http_values),
        "response_ready_ms": summarize(ready_values),
        "validation_overhead_ms": summarize(validation_values),
        "first_token_ms": summarize(first_values),
        "stream_complete_ms": summarize(stream_values),
        "post_first_content_ms": summarize(post_first_values),
        "provider_completion_tokens": summarize(completion_tokens),
        "ttft_ms": summarize(first_values),
        "ttlt_ms": summarize(last_content_values),
        "thinking_estimate_ms": summarize(thinking_estimate_values),
        "composing_estimate_ms": summarize(composing_estimate_values),
        "completion_tail_ms": summarize(completion_tail_values),
        "response_completion_wait_ms": summarize(response_completion_wait_values),
        "usable_ttlt_ms": summarize(usable_ttlt_values),
        "failure_counts": failure_counts,
        "failure_examples": failure_examples,
    }


def make_iteration_result(
    template: PromptTemplate,
    model: ModelConfig,
    iteration: int,
    warmup: bool,
    query: str,
    measurement: RequestMeasurement,
) -> IterationResult:
    return IterationResult(
        schema_version=2,
        template=template.name,
        model=model.deployment,
        display_name=model.display_name,
        iteration=iteration,
        warmup=warmup,
        query=query,
        response_mode=measurement.response_mode,
        success=measurement.success,
        transport_success=measurement.transport_success,
        content_received=measurement.content_received,
        json_parse_valid=measurement.json_parse_valid,
        schema_valid=measurement.schema_valid,
        response_ready=measurement.response_ready,
        http_response_complete_ms=measurement.http_response_complete_ms,
        json_parse_ms=measurement.json_parse_ms,
        schema_validation_ms=measurement.schema_validation_ms,
        response_ready_ms=measurement.response_ready_ms,
        first_token_ms=measurement.first_token_ms,
        last_content_ms=measurement.last_content_ms,
        stream_complete_ms=measurement.stream_complete_ms,
        last_token_ms=measurement.last_content_ms,
        http_status=measurement.http_status,
        failure_phase=measurement.failure_phase,
        error=measurement.error,
        output_preview=measurement.output_preview,
        finish_reason=measurement.finish_reason,
        usage=measurement.usage,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_report_filename(
    audience: str,
    timestamp: time.struct_time | None = None,
) -> str:
    audience_label = {
        "user_to_agent": "user-to-agent",
        "agent_to_agent": "agent-to-agent",
    }[audience]
    timestamp_text = (
        time.strftime("%H%M-%Y%m%d")
        if timestamp is None
        else time.strftime("%H%M-%Y%m%d", timestamp)
    )
    return f"benchmark-{audience_label}-{timestamp_text}.html"


def prepare_reports_dir(out_dir: Path) -> Path:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def prepare_rawdata_dir(out_dir: Path) -> Path:
    rawdata_dir = out_dir / "rawdata"
    rawdata_dir.mkdir(parents=True, exist_ok=True)
    return rawdata_dir


def _run(config_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = prepare_reports_dir(out_dir)
    rawdata_dir = prepare_rawdata_dir(out_dir)
    log_path = rawdata_dir / "benchmark.log"
    raw_path = rawdata_dir / "raw-results.jsonl"
    summary_path = rawdata_dir / "summary.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("benchmark")

    cfg = load_config(config_path)
    report_timestamp = time.localtime()
    user_report_path = reports_dir / build_report_filename("user_to_agent", report_timestamp)
    agent_report_path = reports_dir / build_report_filename("agent_to_agent", report_timestamp)
    templates = [load_template(Path(p)) for p in cfg.templates]

    log.info("Endpoint      : %s", cfg.endpoint)
    log.info("API version   : %s", cfg.api_version)
    log.info("Response mode : streaming")
    log.info("Iterations    : %d measured requests per model/template", cfg.iterations)
    log.info("Models        : %d", len(cfg.models))
    log.info("Templates     : %d (%s)", len(templates), ", ".join(t.name for t in templates))

    log.info("Acquiring Azure token via AzureCliCredential…")
    token, token_tenant_id = acquire_azure_cli_token(cfg.tenant_id)
    log.info("Azure CLI token tenant: %s", token_tenant_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    summary: dict[str, Any] = {
        "schema_version": 2,
        "endpoint": cfg.endpoint,
        "api_version": cfg.api_version,
        "token_budget": cfg.token_budget,
        "iterations": cfg.iterations,
        "response_mode": "streaming",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z").strip(),
        "templates": [],
    }

    with (
        httpx.Client(http2=True) as client,
        raw_path.open("w", encoding="utf-8") as raw_f,
    ):
        for template in templates:
            log.info(
                "=== Template: %s (%d-query pool, %d measured iterations) ===",
                template.name,
                len(template.queries),
                cfg.iterations,
            )
            template_summary: dict[str, Any] = {
                "name": template.name,
                "display_name": template.display_name,
                "query_count": len(template.queries),
                "iterations": cfg.iterations,
                "models": [],
            }
            for model in cfg.models:
                log.info("--- Model: %s (deployment=%s) ---", model.display_name, model.deployment)
                url = build_url(cfg, model.deployment)

                # Warm-up: recorded for audit, excluded from stats.
                warm_query = template.queries[0]
                log.info("  warm-up …")
                warm_measurement = measure_streaming(
                    client,
                    url,
                    headers,
                    build_payload(model, template, warm_query, cfg),
                    cfg.request_timeout_s,
                    template,
                )
                warm_res = make_iteration_result(
                    template, model, 0, True, warm_query, warm_measurement
                )
                raw_f.write(json.dumps(asdict(warm_res)) + "\n")
                raw_f.flush()
                if warm_measurement.response_ready:
                    log.info(
                        "  warm-up ready response_ready=%.0fms",
                        warm_measurement.response_ready_ms or 0,
                    )
                else:
                    log.warning(
                        "  warm-up FAIL phase=%s status=%s err=%s",
                        warm_measurement.failure_phase,
                        warm_measurement.http_status,
                        warm_measurement.error,
                    )

                measured_results: list[IterationResult] = []
                for i, query in iter_benchmark_queries(
                    template.queries,
                    cfg.iterations,
                ):
                    measurement = measure_streaming(
                        client,
                        url,
                        headers,
                        build_payload(model, template, query, cfg),
                        cfg.request_timeout_s,
                        template,
                    )
                    if measurement.response_ready:
                        log.info(
                            "  iter %02d ready response_ready=%.0fms",
                            i,
                            measurement.response_ready_ms or 0,
                        )
                    else:
                        log.warning(
                            "  iter %02d FAIL phase=%s status=%s err=%s",
                            i,
                            measurement.failure_phase,
                            measurement.http_status,
                            measurement.error,
                        )
                    res = make_iteration_result(
                        template, model, i, False, query, measurement
                    )
                    measured_results.append(res)
                    raw_f.write(json.dumps(asdict(res)) + "\n")
                    raw_f.flush()

                template_summary["models"].append(
                    build_model_summary(model, measured_results)
                )
            summary["templates"].append(template_summary)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", summary_path)

    render_report(summary, user_report_path, "user_to_agent")
    render_report(summary, agent_report_path, "agent_to_agent")
    log.info("User report  written to %s", user_report_path)
    log.info("Agent report written to %s", agent_report_path)


# ---------------------------------------------------------------------------
# HTML report renderer (styled to match baseline-v1-report.html)
# ---------------------------------------------------------------------------


REPORT_CSS = (
    ":root{--blue:#155e75;--navy:#19324a;--green:#08783e;--amber:#a15c00;--red:#b42318;"
    "--ink:#172033;--muted:#526079;--line:#dce3ee;--bg:#f4f7fb}"
    "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);"
    "font-family:Aptos,'Trebuchet MS',sans-serif;line-height:1.55}"
    "main{max-width:1450px;margin:auto;padding:24px}"
    "h1{margin:0 0 16px;padding:24px;border-radius:6px;color:#fff;"
    "background:var(--navy);font-size:28px}"
    "h2{color:#29377a;margin-top:28px;border-left:5px solid var(--blue);padding-left:10px}"
    ".subtitle{color:var(--muted);margin:0 0 18px}"
    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}"
    ".card,.section,.chart{background:#fff;border-radius:6px;padding:17px;margin:12px 0;"
    "box-shadow:0 2px 12px #0000000d}"
    ".metric{font-size:25px;font-weight:700;color:var(--blue)}"
    ".ok{color:var(--green);font-weight:700}.warn{color:var(--amber);font-weight:700}"
    ".bad{color:var(--red);font-weight:700}"
    ".scroll{overflow:auto;background:#fff;border-radius:6px;box-shadow:0 2px 12px #0000000d}"
    ".flow{display:flex;align-items:stretch;gap:8px;overflow:auto;padding:8px 0}"
    ".flow-node{min-width:170px;flex:1;padding:14px;border:1px solid var(--line);"
    "border-radius:6px;background:#f8fafc;text-align:center;font-weight:700}"
    ".flow-arrow{align-self:center;color:var(--blue);font-size:24px;font-weight:700}"
    "table{border-collapse:collapse;width:100%;font-size:13px}"
    "th,td{border:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}"
    "th{background:#eaf0ff}th:first-child,td:first-child{text-align:left}"
    "code{background:#eef2ff;padding:2px 5px;border-radius:4px;word-break:break-all}"
    "ul,ol{line-height:1.75}.callout{border-left:5px solid var(--amber);background:#fffaf0}"
    ".footer{font-size:12px;color:var(--muted);text-align:center;margin:28px 0}"
)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.0f} ms"


def render_report(summary: dict[str, Any], out_path: Path, audience: str) -> None:
    if audience not in {"user_to_agent", "agent_to_agent"}:
        raise ValueError(f"unsupported report audience: {audience}")
    total_requests = sum(m["total"] for t in summary["templates"] for m in t["models"])
    total_ready = sum(m["response_ready"] for t in summary["templates"] for m in t["models"])
    validation_failures = sum(
        m["failure_counts"].get("json_parse", 0)
        + m["failure_counts"].get("schema", 0)
        for t in summary["templates"]
        for m in t["models"]
    )
    unique_models = {m["deployment"] for t in summary["templates"] for m in t["models"]}
    report_title = (
        "User-to-Agent Streaming Benchmark Report"
        if audience == "user_to_agent"
        else "Agent-to-Agent Streaming Benchmark Report"
    )
    ranking_metric = (
        "usable_ttlt_ms" if audience == "user_to_agent" else "response_completion_wait_ms"
    )

    def rank_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            models,
            key=lambda model: (
                -model["response_ready_rate"],
                model[ranking_metric]["p95"] if model["response_ready"] else float("inf"),
                model[ranking_metric]["mean"] if model["response_ready"] else float("inf"),
            ),
        )

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append(f"<title>{report_title}</title>")
    parts.append('<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>')
    parts.append(f"<style>{REPORT_CSS}</style></head><body><main>")

    parts.append(f"<h1>{report_title}</h1>")
    if summary.get("illustrative"):
        parts.append(
            '<div class="section callout"><b>Illustrative sample data.</b> '
            "These values are deterministic examples, not live Azure benchmark results.</div>"
        )
    parts.append(
        f'<p class="subtitle">Endpoint <code>{_esc(summary["endpoint"])}</code> · '
        f'API {_esc(summary["api_version"])} · Generated {_esc(summary["generated_at"])}</p>'
    )

    parts.append('<div class="grid">')
    parts.append(
        f'<div class="card"><b>Measured workload</b><div class="metric">{total_requests} requests</div>'
        f'{len(unique_models)} models × {len(summary["templates"])} templates</div>'
    )
    parts.append(
        f'<div class="card"><b>Response ready</b><div class="metric">{total_ready}/{total_requests}</div>'
        f'{(100 * total_ready / total_requests if total_requests else 0):.1f}% structurally usable</div>'
    )
    parts.append(
        '<div class="card"><b>Response mode</b><div class="metric">streaming</div>'
        "Both reports use the same streaming measurements</div>"
    )
    parts.append(
        f'<div class="card"><b>Validation failures</b><div class="metric bad">{validation_failures}</div>'
        "Assistant JSON parse or schema failures</div>"
    )
    parts.append(
        '<div class="card"><b>Ranking scope</b><div class="metric">Per template</div>'
        "Unlike workloads are never combined into one winner</div>"
    )
    parts.append("</div>")

    parts.append('<h2>1. Measurement flow</h2><div class="section">')
    if audience == "user_to_agent":
        flow_nodes = [
            "Send API request",
            "First token arrives (TTFT)",
            "Last token arrives (TTLT)",
            "Answer complete",
        ]
    else:
        flow_nodes = [
            "Agent A sends request",
            "First token arrives; Agent B waits",
            "Last token arrives",
            "Complete response received",
            "Agent B starts",
        ]
    parts.append('<div class="flow">')
    for flow_index, flow_node in enumerate(flow_nodes):
        if flow_index:
            parts.append('<div class="flow-arrow" aria-hidden="true">&#8594;</div>')
        parts.append(f'<div class="flow-node">{_esc(flow_node)}</div>')
    parts.append("</div></div>")

    parts.append("<h2>2. Project setting</h2>")
    parts.append('<div class="section"><table>')
    parts.append("<tr><th>Item</th><th>Configuration</th></tr>")
    parts.append(f'<tr><td>Endpoint</td><td><code>{_esc(summary["endpoint"])}</code></td></tr>')
    parts.append(f'<tr><td>API version</td><td>{_esc(summary["api_version"])}</td></tr>')
    parts.append("<tr><td>Authentication</td><td>Azure CLI via AzureCliCredential; token cached before per-request timer</td></tr>")
    parts.append(
        f"<tr><td>API</td><td>Chat Completions; HTTP/2 enabled when negotiated "
        "(negotiated protocol not recorded); response mode: streaming</td></tr>"
    )
    parts.append("<tr><td>Execution</td><td>Concurrency 1; sequential per model; persistent HTTP client; one warm-up call per (template, model) recorded but excluded from stats</td></tr>")
    parts.append(f'<tr><td>Output limit</td><td>{summary["token_budget"]:,} tokens (sent as <code>max_completion_tokens</code> for reasoning models, otherwise <code>max_tokens</code>)</td></tr>')
    configured_iterations = summary.get("iterations")
    if configured_iterations is not None:
        parts.append(
            f"<tr><td>Measured iterations</td><td>{configured_iterations} per model/template, "
            "plus one excluded warm-up</td></tr>"
        )
    parts.append(
        "<tr><td>Templates</td><td>"
        + ", ".join(_esc(t["display_name"]) for t in summary["templates"])
        + "</td></tr>"
    )
    parts.append("</table></div>")

    parts.append("<h2>3. Measurement contract</h2>")
    parts.append('<div class="section">')
    parts.append(
        "<p><b>This is not isolated inference time.</b> Client-observed latency includes network "
        "transport, Azure front-end processing, queueing, prompt ingestion, model work, and response "
        "transfer. Authentication occurs before the per-request timer.</p>"
    )
    parts.append(
        "<p><b>Thinking estimate</b> is TTFT: request start to first visible content. "
        "<b>Composing estimate</b> is TTLT minus TTFT: first to final visible content.</p>"
    )
    if audience == "user_to_agent":
        parts.append(
            "<p>The stacked timing total is <b>TTLT</b>. TTFT is both the first milestone and the "
            "thinking segment; it is not added a second time.</p>"
        )
    else:
        parts.append(
            "<p>The next agent starts only after the complete response arrives. The stacked timing "
            "total is the <b>response completion wait</b>; TTFT alone is not the primary result.</p>"
        )
    parts.append(
        "<p>Measured iterations select queries in template order and cycle through the query pool "
        "when necessary. Reported statistics "
        "focus on average latency and slow-case P95 over response-ready iterations. The warm-up call "
        "is recorded in <code>rawdata/raw-results.jsonl</code> for audit but excluded from stats.</p>"
    )
    parts.append("</div>")

    for idx, t in enumerate(summary["templates"], start=4):
        measured_iterations = t.get("iterations", t["query_count"])
        parts.append(f'<h2>{idx}. Results — {_esc(t["display_name"])}</h2>')
        parts.append(
            f'<div class="section"><p>Template <code>{_esc(t["name"])}</code> has a pool of '
            f"{t['query_count']} user queries. Each model runs one warm-up (discarded) then "
            f"{measured_iterations} timed "
            f"iterations. Rows rank by usable rate, then {('TTLT' if audience == 'user_to_agent' else 'completion-wait')} "
            "P95, then average.</p></div>"
        )
        chart_id = f"chart_{idx}"
        parts.append(f'<div class="chart"><canvas id="{chart_id}"></canvas></div>')
        parts.append('<div class="scroll"><table><thead><tr>')
        parts.append("<th>Model</th><th>Usable</th><th>Usable rate</th>")
        if audience == "user_to_agent":
            parts.append(
                "<th>Average TTFT / thinking estimate</th>"
                "<th>Average composing estimate</th><th>Average TTLT</th>"
                "<th>Slow-case TTLT (P95)</th>"
            )
        else:
            parts.append(
                "<th>Average thinking estimate</th><th>Average composing estimate</th>"
                "<th>Average response completion wait</th>"
                "<th>Slow-case completion wait (P95)</th>"
            )
        parts.append("</tr></thead><tbody>")
        rows = rank_models(t["models"])
        for m in rows:
            success_cls = (
                "ok"
                if m["response_ready"] == m["total"]
                else ("warn" if m["response_ready"] > 0 else "bad")
            )
            parts.append(f'<tr><td>{_esc(m["display_name"])}</td>')
            parts.append(f'<td class="{success_cls}">{m["response_ready"]}/{m["total"]}</td>')
            parts.append(f'<td>{100 * m["response_ready_rate"]:.1f}%</td>')
            if m["response_ready"] == 0:
                parts.extend(["<td>—</td>"] * 4)
            elif audience == "user_to_agent":
                for metric, statistic in (
                    ("thinking_estimate_ms", "mean"),
                    ("composing_estimate_ms", "mean"),
                    ("usable_ttlt_ms", "mean"),
                    ("usable_ttlt_ms", "p95"),
                ):
                    parts.append(f'<td>{_fmt_ms(m[metric][statistic])}</td>')
            else:
                for metric, statistic in (
                    ("thinking_estimate_ms", "mean"),
                    ("composing_estimate_ms", "mean"),
                    ("response_completion_wait_ms", "mean"),
                    ("response_completion_wait_ms", "p95"),
                ):
                    parts.append(f'<td>{_fmt_ms(m[metric][statistic])}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
        parts.append('<p class="subtitle">Latency values are client-observed and shown in ms, or seconds when ≥ 1000 ms.</p>')

        failures = [
            (
                model["display_name"],
                phase,
                count,
                model["failure_examples"].get(phase, ""),
            )
            for model in t["models"]
            for phase, count in model["failure_counts"].items()
        ]
        if failures:
            parts.append("<h3>Failure phases</h3><div class=\"scroll\"><table>")
            parts.append("<thead><tr><th>Model</th><th>Phase</th><th>Count</th><th>Example</th></tr></thead><tbody>")
            for model_name, phase, count, example in failures:
                parts.append(
                    f"<tr><td>{_esc(model_name)}</td><td>{_esc(phase)}</td><td>{count}</td>"
                    f"<td>{_esc(example)}</td></tr>"
                )
            parts.append("</tbody></table></div>")

    final_idx = len(summary["templates"]) + 3
    parts.append(f"<h2>{final_idx}. Customizing the sample templates</h2>")
    parts.append('<div class="section callout">')
    parts.append("<p>Edit the YAML template files to model your own workload:</p>")
    parts.append("<ul>")
    for t in summary["templates"]:
        parts.append(
            f'<li><code>prompt-template-{_esc(t["name"])}.yaml</code> — {_esc(t["display_name"])}</li>'
        )
    parts.append("</ul>")
    parts.append(
        "<p>Templates define metadata, request controls, the system prompt, queries, and an optional "
        "<code>validation.schema</code>. A response is ready only after it satisfies the configured "
        "schema. Structural validation does not evaluate semantic correctness.</p>"
    )
    parts.append("</div>")

    parts.append(
        '<div class="footer">Source artifacts: rawdata/raw-results.jsonl, rawdata/summary.json, '
        "rawdata/benchmark.log, benchmark.py. "
        "Response-ready latency is client-observed end-to-end time, not isolated model inference.</div>"
    )
    parts.append("</main>")

    parts.append("<script>")
    for idx, t in enumerate(summary["templates"], start=4):
        rows = rank_models(t["models"])
        labels = [m["display_name"] for m in rows]
        thinking = [round(m["thinking_estimate_ms"]["mean"], 1) for m in rows]
        composing = [round(m["composing_estimate_ms"]["mean"], 1) for m in rows]
        if audience == "user_to_agent":
            datasets = [
                {"label": "Thinking estimate (TTFT)", "data": thinking, "backgroundColor": "#155e75", "stack": "timing"},
                {"label": "Composing estimate (TTLT - TTFT)", "data": composing, "backgroundColor": "#d97706", "stack": "timing"},
                {"type": "line", "label": "Average TTLT marker", "data": [round(m["usable_ttlt_ms"]["mean"], 1) for m in rows], "borderColor": "#19324a", "backgroundColor": "#19324a"},
            ]
            chart_title = t["display_name"] + " - stacked total is TTLT (ms)"
        else:
            datasets = [
                {"label": "Thinking estimate (TTFT)", "data": thinking, "backgroundColor": "#155e75", "stack": "timing"},
                {"label": "Composing estimate (TTLT - TTFT)", "data": composing, "backgroundColor": "#d97706", "stack": "timing"},
                {"label": "Completion tail", "data": [round(m["completion_tail_ms"]["mean"], 1) for m in rows], "backgroundColor": "#64748b", "stack": "timing"},
                {"type": "line", "label": "Response completion wait marker", "data": [round(m["response_completion_wait_ms"]["mean"], 1) for m in rows], "borderColor": "#19324a", "backgroundColor": "#19324a"},
            ]
            chart_title = t["display_name"] + " - stacked total is response completion wait (ms)"
        parts.append(
            f'new Chart(document.getElementById("chart_{idx}"),{{type:"bar",'
            f'data:{{labels:{json.dumps(labels)},'
            f'datasets:{json.dumps(datasets)}}},'
            f'options:{{responsive:true,plugins:{{legend:{{position:"top"}},'
            f'title:{{display:true,text:{json.dumps(chart_title)}}}}},'
            f'scales:{{x:{{stacked:true}},y:{{stacked:true,beginAtZero:true,title:{{display:true,text:"Milliseconds"}}}}}}}}}});'
        )
    parts.append("</script></body></html>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def render_summary_reports(
    summary_path: Path,
    out_dir: Path,
    timestamp: time.struct_time | None = None,
) -> tuple[Path, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_timestamp = timestamp or time.localtime()
    reports_dir = prepare_reports_dir(out_dir)
    user_report_path = reports_dir / build_report_filename("user_to_agent", report_timestamp)
    agent_report_path = reports_dir / build_report_filename("agent_to_agent", report_timestamp)
    render_report(summary, user_report_path, "user_to_agent")
    render_report(summary, agent_report_path, "agent_to_agent")
    return user_report_path, agent_report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Azure OpenAI response-ready benchmark")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--out", default=".", help="Output directory for artifacts")
    ap.add_argument(
        "--render-summary",
        help="Render both audience reports from an existing summary JSON without calling Azure",
    )
    ap.add_argument(
        "--report-timestamp",
        help="Optional fixed report timestamp in HHMM-YYYYMMDD format",
    )
    args = ap.parse_args()
    if args.render_summary:
        timestamp = (
            time.strptime(args.report_timestamp, "%H%M-%Y%m%d")
            if args.report_timestamp
            else None
        )
        render_summary_reports(Path(args.render_summary), Path(args.out), timestamp)
        return
    if args.report_timestamp:
        ap.error("--report-timestamp requires --render-summary")
    _run(Path(args.config), Path(args.out))


if __name__ == "__main__":
    main()
