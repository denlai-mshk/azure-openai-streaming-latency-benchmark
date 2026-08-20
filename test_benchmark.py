import base64
import json
import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import benchmark


def make_token(tenant_id: str) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"tid": tenant_id}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"header.{payload}.signature"


class FakeAccessToken:
    def __init__(self, token: str) -> None:
        self.token = token


class FakeCredential:
    token = ""
    requested_scope = ""
    constructed_with: dict[str, str] = {}

    def __init__(self, **kwargs: str) -> None:
        type(self).constructed_with = kwargs

    def get_token(self, scope: str) -> FakeAccessToken:
        type(self).requested_scope = scope
        return FakeAccessToken(type(self).token)


class AzureCliAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCredential.requested_scope = ""
        FakeCredential.constructed_with = {}

    @patch.object(benchmark, "AzureCliCredential", FakeCredential)
    def test_uses_cli_credential_tenant_and_cognitive_services_scope(self) -> None:
        tenant_id = "11111111-1111-1111-1111-111111111111"
        FakeCredential.token = make_token(tenant_id)

        token, actual_tenant_id = benchmark.acquire_azure_cli_token(tenant_id)

        self.assertEqual(token, FakeCredential.token)
        self.assertEqual(actual_tenant_id, tenant_id)
        self.assertEqual(FakeCredential.constructed_with, {"tenant_id": tenant_id})
        self.assertEqual(FakeCredential.requested_scope, benchmark.AZURE_OPENAI_SCOPE)

    @patch.object(benchmark, "AzureCliCredential", FakeCredential)
    def test_rejects_token_from_another_tenant(self) -> None:
        FakeCredential.token = make_token("wrong-tenant")

        with self.assertRaisesRegex(RuntimeError, "does not match configured tenant_id"):
            benchmark.acquire_azure_cli_token("expected-tenant")

    @patch.object(benchmark, "AzureCliCredential", FakeCredential)
    def test_rejects_token_without_readable_tenant_claim(self) -> None:
        FakeCredential.token = "not-a-jwt"

        with self.assertRaisesRegex(RuntimeError, "readable tenant claim"):
            benchmark.acquire_azure_cli_token(None)

    @patch.object(benchmark, "AzureCliCredential", FakeCredential)
    def test_does_not_log_token(self) -> None:
        tenant_id = "22222222-2222-2222-2222-222222222222"
        FakeCredential.token = make_token(tenant_id)

        with self.assertLogs(level=logging.DEBUG) as captured:
            logging.getLogger().debug("authentication test")
            benchmark.acquire_azure_cli_token(tenant_id)

        self.assertNotIn(FakeCredential.token, "\n".join(captured.output))


class RequestPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parent
        cls.routing = benchmark.load_template(root / "prompt-template-routing.yaml")
        cls.reasoning = benchmark.load_template(root / "prompt-template-reasoning.yaml")
        cls.config = benchmark.BenchmarkConfig(
            endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
            tenant_id=None,
            token_budget=2000,
            request_timeout_s=180,
            models=[],
            templates=[],
        )

    def test_loads_template_request_defaults(self) -> None:
        self.assertEqual(self.routing.reasoning_effort, "low")
        self.assertEqual(self.reasoning.reasoning_effort, "medium")
        for template in (self.routing, self.reasoning):
            self.assertEqual(template.temperature, 0)
            self.assertEqual(template.top_p, 1)
            self.assertIsNotNone(template.validation)

    def test_reasoning_model_uses_template_effort_without_sampling_controls(self) -> None:
        model = benchmark.ModelConfig(
            deployment="gpt-5",
            display_name="gpt-5",
            reasoning_effort="high",
            supports_reasoning_effort=True,
        )

        routing_payload = benchmark.build_payload(
            model, self.routing, "route this", self.config
        )
        reasoning_payload = benchmark.build_payload(
            model, self.reasoning, "solve this", self.config
        )

        self.assertEqual(routing_payload["reasoning_effort"], "low")
        self.assertEqual(reasoning_payload["reasoning_effort"], "medium")
        for payload in (routing_payload, reasoning_payload):
            self.assertEqual(payload["max_completion_tokens"], 2000)
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("temperature", payload)
            self.assertNotIn("top_p", payload)

    def test_non_reasoning_model_uses_deterministic_sampling(self) -> None:
        model = benchmark.ModelConfig(
            deployment="gpt-4.1",
            display_name="gpt-4.1",
        )

        for template in (self.routing, self.reasoning):
            with self.subTest(template=template.name):
                payload = benchmark.build_payload(
                    model, template, "answer this", self.config
                )
                self.assertEqual(payload["max_tokens"], 2000)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["top_p"], 1)
                self.assertNotIn("max_completion_tokens", payload)
                self.assertNotIn("reasoning_effort", payload)

    def test_payload_always_uses_streaming(self) -> None:
        model = benchmark.ModelConfig("gpt-4.1", "gpt-4.1")
        payload = benchmark.build_payload(
            model, self.routing, "route this", self.config
        )
        self.assertTrue(payload["stream"])

    def test_legacy_reasoning_effort_implies_capability(self) -> None:
        config_yaml = """\
endpoint: https://example.openai.azure.com
api_version: 2024-10-21
token_budget: 100
models:
  - deployment: legacy-gpt-5
    display_name: Legacy GPT-5
    reasoning_effort: high
templates: []
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(config_yaml, encoding="utf-8")
            model = benchmark.load_config(path).models[0]

        self.assertTrue(model.supports_reasoning_effort)
        self.assertEqual(model.reasoning_effort, "high")

    def test_iterations_default_to_ten_and_accept_positive_override(self) -> None:
        config_yaml = """\
endpoint: https://example.openai.azure.com
api_version: 2024-10-21
models: []
templates: []
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(config_yaml, encoding="utf-8")
            self.assertEqual(benchmark.load_config(path).iterations, 10)

            path.write_text(config_yaml + "iterations: 12\n", encoding="utf-8")
            self.assertEqual(benchmark.load_config(path).iterations, 12)

    def test_iterations_reject_invalid_values(self) -> None:
        config_yaml = """\
endpoint: https://example.openai.azure.com
api_version: 2024-10-21
models: []
templates: []
iterations: VALUE
"""
        for value in ("true", "0", "-1", "1.5", '"10"'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "config.yaml"
                path.write_text(config_yaml.replace("VALUE", value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "iterations must be an integer"):
                    benchmark.load_config(path)

    def test_query_selection_cycles_to_configured_iteration_count(self) -> None:
        queries = [f"query-{index}" for index in range(1, 11)]

        selected = list(benchmark.iter_benchmark_queries(queries, 12))

        self.assertEqual([iteration for iteration, _ in selected], list(range(1, 13)))
        self.assertEqual(
            [query for _, query in selected],
            queries + ["query-1", "query-2"],
        )


class StreamingMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = benchmark.load_template(
            Path(__file__).parent / "prompt-template-routing.yaml"
        )
        cls.valid_content = json.dumps({
            "intent": "billing",
            "agent": "billing_agent",
            "confidence": 0.9,
            "reason": "Invoice question.",
        })

    def measure(self, include_done: bool) -> benchmark.RequestMeasurement:
        event = {
            "choices": [{
                "delta": {"content": self.valid_content},
                "finish_reason": "stop",
            }]
        }
        body = f"data: {json.dumps(event)}\n\n"
        if include_done:
            body += "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=body.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return benchmark.measure_streaming(
                client,
                "https://example.test/chat/completions",
                {},
                {"stream": True},
                10,
                self.template,
            )

    def test_valid_stream_is_ready_after_done(self) -> None:
        result = self.measure(include_done=True)

        self.assertTrue(result.response_ready)
        self.assertTrue(result.schema_valid)
        self.assertIsNotNone(result.first_token_ms)
        self.assertIsNotNone(result.last_content_ms)
        self.assertIsNotNone(result.stream_complete_ms)
        self.assertIsNone(result.http_response_complete_ms)

        iteration = benchmark.make_iteration_result(
            self.template,
            benchmark.ModelConfig("gpt-4.1", "gpt-4.1"),
            1,
            False,
            "query",
            result,
        )
        self.assertEqual(iteration.last_token_ms, iteration.last_content_ms)
        summary = benchmark.build_model_summary(
            benchmark.ModelConfig("gpt-4.1", "gpt-4.1"),
            [iteration],
        )
        self.assertEqual(summary["ttlt_ms"]["p95"], iteration.last_content_ms)
        self.assertEqual(
            summary["stream_complete_ms"]["p95"],
            iteration.stream_complete_ms,
        )

    def test_stream_without_done_is_not_ready(self) -> None:
        result = self.measure(include_done=False)

        self.assertFalse(result.response_ready)
        self.assertEqual(result.failure_phase, "stream_terminal")

    def test_template_without_validation_does_not_claim_json_validity(self) -> None:
        template = benchmark.PromptTemplate("freeform", "Freeform", "system", [])

        result = benchmark.parse_and_validate_output("plain text", template)

        self.assertIsNone(result.json_parse_valid)
        self.assertIsNone(result.schema_valid)


class SummaryAndReportTests(unittest.TestCase):
    def test_output_subdirectories_are_created_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "output"

            reports_dir = benchmark.prepare_reports_dir(out_dir)
            rawdata_dir = benchmark.prepare_rawdata_dir(out_dir)

            self.assertEqual(reports_dir, out_dir / "reports")
            self.assertEqual(rawdata_dir, out_dir / "rawdata")
            self.assertTrue(reports_dir.is_dir())
            self.assertTrue(rawdata_dir.is_dir())

    def test_report_filenames_identify_audience_and_shared_run_time(self) -> None:
        timestamp = time.struct_time((2026, 8, 20, 14, 30, 0, 3, 232, -1))

        self.assertEqual(
            benchmark.build_report_filename("user_to_agent", timestamp),
            "benchmark-user-to-agent-1430-20260820.html",
        )
        self.assertEqual(
            benchmark.build_report_filename("agent_to_agent", timestamp),
            "benchmark-agent-to-agent-1430-20260820.html",
        )

    def make_result(
        self,
        model: benchmark.ModelConfig,
        first_token_ms: float | None,
        last_content_ms: float | None = None,
        stream_complete_ms: float | None = None,
        failure_phase: str | None = None,
    ) -> benchmark.IterationResult:
        ready = failure_phase is None and stream_complete_ms is not None
        measurement = benchmark.RequestMeasurement(
            response_mode="streaming",
            success=ready,
            transport_success=True,
            content_received=True,
            json_parse_valid=ready,
            schema_valid=ready,
            response_ready=ready,
            http_response_complete_ms=None,
            json_parse_ms=0.2,
            schema_validation_ms=0.3,
            response_ready_ms=(stream_complete_ms + 0.5 if ready else None),
            first_token_ms=first_token_ms,
            last_content_ms=last_content_ms,
            stream_complete_ms=stream_complete_ms,
            http_status=200,
            failure_phase=failure_phase,
            error="invalid output" if failure_phase else None,
            output_preview="{}",
            finish_reason="stop",
            usage=None,
        )
        template = benchmark.PromptTemplate("routing", "Routing", "system", [])
        return benchmark.make_iteration_result(
            template, model, 1, False, "query", measurement
        )

    def make_summary(self, models: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": 2,
            "endpoint": "https://example.test",
            "api_version": "2024-10-21",
            "token_budget": 100,
            "response_mode": "streaming",
            "generated_at": "2026-08-20",
            "templates": [{
                "name": "routing",
                "display_name": "Routing",
                "query_count": 2,
                "models": models,
            }],
        }

    def test_summary_segments_are_additive_and_ready_only(self) -> None:
        model = benchmark.ModelConfig("deployment", "Model")
        summary = benchmark.build_model_summary(
            model,
            [
                self.make_result(model, 4000, 9000, 10000),
                self.make_result(model, 100, 200, 300, "schema"),
            ],
        )

        self.assertEqual(summary["response_ready"], 1)
        self.assertEqual(summary["response_ready_rate"], 0.5)
        self.assertEqual(summary["thinking_estimate_ms"]["mean"], 4000)
        self.assertEqual(summary["composing_estimate_ms"]["mean"], 5000)
        self.assertEqual(summary["completion_tail_ms"]["mean"], 1000)
        self.assertEqual(summary["response_completion_wait_ms"]["mean"], 10000)
        self.assertEqual(
            summary["thinking_estimate_ms"]["mean"]
            + summary["composing_estimate_ms"]["mean"],
            summary["usable_ttlt_ms"]["mean"],
        )
        self.assertEqual(
            summary["thinking_estimate_ms"]["mean"]
            + summary["composing_estimate_ms"]["mean"]
            + summary["completion_tail_ms"]["mean"],
            summary["response_completion_wait_ms"]["mean"],
        )
        self.assertEqual(summary["failure_counts"], {"schema": 1})

    def test_user_report_uses_ttlt_presentation_and_ranking(self) -> None:
        slower = benchmark.ModelConfig("slower", "Slower TTLT")
        faster = benchmark.ModelConfig("faster", "Faster TTLT")
        summary = self.make_summary([
            benchmark.build_model_summary(slower, [self.make_result(slower, 10, 80, 90)]),
            benchmark.build_model_summary(faster, [self.make_result(faster, 20, 60, 100)]),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.html"
            benchmark.render_report(summary, report_path, "user_to_agent")
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("User-to-Agent Streaming Benchmark Report", report)
        self.assertIn("First token arrives (TTFT)", report)
        self.assertIn("Average TTFT / thinking estimate", report)
        self.assertIn("Composing estimate (TTLT - TTFT)", report)
        self.assertIn("Average TTLT marker", report)
        self.assertNotIn("Completion tail", report)
        self.assertLess(report.index("Faster TTLT"), report.index("Slower TTLT"))

    def test_agent_report_uses_completion_wait_presentation_and_ranking(self) -> None:
        slower = benchmark.ModelConfig("slower", "Slower completion")
        faster = benchmark.ModelConfig("faster", "Faster completion")
        summary = self.make_summary([
            benchmark.build_model_summary(slower, [self.make_result(slower, 10, 50, 100)]),
            benchmark.build_model_summary(faster, [self.make_result(faster, 20, 60, 80)]),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.html"
            benchmark.render_report(summary, report_path, "agent_to_agent")
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("Agent-to-Agent Streaming Benchmark Report", report)
        self.assertIn("Agent B starts", report)
        self.assertIn("Average response completion wait", report)
        self.assertIn("Completion tail", report)
        self.assertIn("Response completion wait marker", report)
        self.assertLess(report.index("Faster completion"), report.index("Slower completion"))

    def test_offline_render_writes_both_reports_without_authentication(self) -> None:
        model = benchmark.ModelConfig("sample", "Sample model")
        summary = self.make_summary([
            benchmark.build_model_summary(model, [self.make_result(model, 40, 90, 100)])
        ])
        timestamp = time.struct_time((2026, 8, 20, 12, 0, 0, 3, 232, -1))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            summary_path = temp_path / "sample.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            paths = benchmark.render_summary_reports(summary_path, temp_path, timestamp)

            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(all(path.parent == temp_path / "reports" for path in paths))
            self.assertEqual(paths[0].name, "benchmark-user-to-agent-1200-20260820.html")
            self.assertEqual(paths[1].name, "benchmark-agent-to-agent-1200-20260820.html")


if __name__ == "__main__":
    unittest.main()