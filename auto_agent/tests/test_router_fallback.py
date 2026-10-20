"""Deterministic tests for provider fallback and persisted router state."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
import types
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from core import llm_helper
from core import router


def _http_error(status, message=""):
    response = mock.Mock(status_code=status)
    return requests.HTTPError(message, response=response)


class RouterFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp_dir.name) / "router_state.json"
        self.state_path_patch = mock.patch.object(
            llm_helper, "ROUTER_STATE_PATH", self.state_path
        )
        self.state_path_patch.start()
        self.catalog_path_patch = mock.patch.object(
            llm_helper,
            "OPENROUTER_MODELS_CACHE_PATH",
            Path(self.temp_dir.name) / "openrouter_models.json",
        )
        self.catalog_path_patch.start()
        llm_helper.reset_router_state()
        self.key_patches = [
            mock.patch.object(llm_helper, "OLLAMA_PREFER_LOCAL", False),
            mock.patch.object(llm_helper, "KILO_API_KEY", "kilo-test"),
            mock.patch.object(llm_helper, "OPENROUTER_API_KEY", "openrouter-test"),
            mock.patch.object(
                llm_helper,
                "OPENROUTER_FREE_POOL",
                ("openrouter-primary", "openrouter-secondary"),
            ),
            mock.patch.object(
                llm_helper, "OPENROUTER_PRIMARY_MODEL", "openrouter-primary"
            ),
            mock.patch.object(
                llm_helper, "OPENROUTER_FALLBACK_MODEL", "openrouter-fallback"
            ),
            mock.patch.object(
                llm_helper, "openrouter_model_is_available", return_value=(True, "")
            ),
            mock.patch.object(llm_helper, "GEMINI_API_KEY", "gemini-test"),
            mock.patch.object(llm_helper, "GROQ_API_KEY", "groq-test"),
            mock.patch.object(
                llm_helper, "_record_provider_event", mock.Mock()
            ),
        ]
        for patcher in self.key_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.key_patches):
            patcher.stop()
        self.catalog_path_patch.stop()
        self.state_path_patch.stop()
        self.temp_dir.cleanup()

    def test_swarm_router_prompt_disallows_physical_desktop(self):
        client = mock.Mock()
        client.ask.return_value = {"tool": "browser_tool", "action": "open"}
        with (
            mock.patch.object(
                router.SkillResolver,
                "resolve",
                return_value={"skill": "browser_tool"},
            ),
            mock.patch.object(
                router,
                "get_all_skills",
                return_value={"browser_tool": {"description": "browser"}},
            ),
        ):
            router.route_task(
                {"desc": "open a web page"}, "", llm_client=client,
                swarm_agent_id="agent-1",
            )

        prompt = client.ask.call_args.args[0]
        self.assertIn("Không dùng computer_tool", prompt)
        self.assertIn("đang bị vô hiệu hóa cho agent", prompt)
        self.assertNotIn("computer_tool.perform", prompt)
        self.assertNotIn("coder_tool", prompt)

    def test_tier1_fail_tier2_ok(self):
        with (
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=_http_error(429)
            ),
            mock.patch.object(llm_helper, "_call_kilo", return_value="hello from Kilo"),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from Kilo")
        self.assertEqual(llm_helper._ROUTER_STATE["last_good_tier"], 2)

    def test_strict_model_api_check_does_not_hide_failure_with_fallback(self):
        with (
            mock.patch.object(
                llm_helper, "_call_kilo", side_effect=_http_error(429)
            ) as kilo,
            mock.patch.object(
                llm_helper, "_call_openrouter", side_effect=AssertionError
            ) as openrouter,
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=AssertionError
            ) as groq,
        ):
            with self.assertRaisesRegex(RuntimeError, "All LLM tiers failed"):
                llm_helper.call_llm(
                    "Reply OK", model="kilo:kilo-auto/free",
                    max_output_tokens=8, allow_fallback=False,
                )

        kilo.assert_called_once()
        openrouter.assert_not_called()
        groq.assert_not_called()

    def test_extract_json_from_surrounding_prose_and_markdown(self):
        parsed = llm_helper._extract_json(
            'Here is the plan:\n```JSON\n{"tasks": []}\n```\nDone.'
        )

        self.assertEqual(parsed, {"tasks": []})

    def test_groq_retries_json_request_without_response_format_on_400(self):
        response = mock.Mock()
        response.choices = [
            mock.Mock(message=mock.Mock(content='{"tasks": []}'))
        ]
        client = mock.Mock()
        client.chat.completions.create.side_effect = [
            _http_error(400, "response_format is not supported"),
            response,
        ]
        fake_groq = types.SimpleNamespace(
            Groq=mock.Mock(return_value=client)
        )
        with mock.patch.dict(sys.modules, {"groq": fake_groq}):
            result = llm_helper._call_groq("make a plan", "test-model", True)

        self.assertEqual(result, '{"tasks": []}')
        self.assertEqual(client.chat.completions.create.call_count, 2)
        first_request = client.chat.completions.create.call_args_list[0].kwargs
        retry_request = client.chat.completions.create.call_args_list[1].kwargs
        self.assertIn("response_format", first_request)
        self.assertNotIn("response_format", retry_request)

    def test_groq_does_not_retry_unrelated_bad_request(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = _http_error(
            400, "model is not available"
        )
        fake_groq = types.SimpleNamespace(
            Groq=mock.Mock(return_value=client)
        )
        with mock.patch.dict(sys.modules, {"groq": fake_groq}):
            with self.assertRaises(requests.HTTPError):
                llm_helper._call_groq("make a plan", "test-model", True)

        client.chat.completions.create.assert_called_once()

    def test_gemini_passes_generation_config_to_supported_parameter(self):
        response = mock.Mock(text='{"tasks": []}')
        client = mock.Mock()
        client.models.generate_content.return_value = response
        fake_genai = types.SimpleNamespace(
            Client=mock.Mock(return_value=client),
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        with mock.patch.dict(
            sys.modules,
            {"google": fake_google, "google.genai": fake_genai},
        ):
            result = llm_helper._call_gemini("make a plan", "test-model", True)

        self.assertEqual(result, '{"tasks": []}')
        fake_genai.Client.assert_called_once_with(
            api_key=llm_helper.GEMINI_API_KEY,
            http_options={"timeout": llm_helper.REQUEST_TIMEOUT * 1000},
        )
        client.models.generate_content.assert_called_once_with(
            model="test-model",
            contents="make a plan",
            config={
                "max_output_tokens": llm_helper._max_output_tokens(True),
                "response_mime_type": "application/json",
            },
        )

    def test_gemini_accepts_a_larger_output_budget_for_slide_drafts(self):
        response = mock.Mock(text="draft")
        client = mock.Mock()
        client.models.generate_content.return_value = response
        fake_genai = types.SimpleNamespace(Client=mock.Mock(return_value=client))
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        with mock.patch.dict(
            sys.modules,
            {"google": fake_google, "google.genai": fake_genai},
        ):
            result = llm_helper._call_gemini(
                "slide prompt", "test-model", False, max_output_tokens=3072
            )

        self.assertEqual(result, "draft")
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["config"],
            {"max_output_tokens": 3072},
        )

    def test_call_llm_passes_explicit_output_budget_to_preferred_free_provider(self):
        with (
            mock.patch.object(llm_helper, "GEMINI_API_KEY", "test-key"),
            mock.patch.object(llm_helper, "OLLAMA_PREFER_LOCAL", False),
            mock.patch.object(
                llm_helper, "_call_gemini", return_value="complete draft"
            ) as gemini,
            mock.patch.object(llm_helper, "_call_groq", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_kilo", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_openrouter", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_ollama", side_effect=AssertionError),
        ):
            result = llm_helper.call_llm(
                "slide prompt", max_output_tokens=3072
            )

        self.assertEqual(result, "complete draft")
        gemini.assert_called_once_with(
            "slide prompt", llm_helper.GEMINI_MODEL, False, 3072
        )

    def test_auto_route_prefers_free_long_context_gemini(self):
        with (
            mock.patch.object(llm_helper, "OLLAMA_PREFER_LOCAL", False),
            mock.patch.object(llm_helper, "_call_gemini", return_value="gemini result") as gemini,
            mock.patch.object(llm_helper, "_call_groq", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_kilo", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_openrouter", side_effect=AssertionError),
            mock.patch.object(llm_helper, "_call_ollama", side_effect=AssertionError),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "gemini result")
        gemini.assert_called_once_with(
            "hello", llm_helper.GEMINI_MODEL, False
        )

    def test_ollama_model_probe_uses_local_tags_api(self):
        response = mock.Mock()
        response.json.return_value = {
            "models": [{"name": "qwen2.5-coder:1.5b"}]
        }
        with mock.patch.object(
            llm_helper.requests, "get", return_value=response
        ) as get:
            installed, reason = llm_helper.ollama_model_is_installed(
                "qwen2.5-coder:1.5b"
            )

        self.assertTrue(installed)
        self.assertEqual(reason, "")
        get.assert_called_once_with(
            f"{llm_helper.OLLAMA_HOST.rstrip('/')}/api/tags",
            timeout=10,
        )

    def test_ollama_model_discovery_collapses_aliases_and_prefers_configured_model(self):
        response = mock.Mock()
        response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:1.5b.bak", "digest": "same"},
                {"name": "qwen2.5:3b", "digest": "three-b"},
                {"name": "qwen2.5-coder:1.5b", "digest": "same"},
                {"name": "qwen2.5:7b", "digest": "seven-b"},
            ]
        }
        with (
            mock.patch.object(llm_helper, "OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
            mock.patch.object(llm_helper.requests, "get", return_value=response),
        ):
            models = llm_helper.ollama_installed_models()

        self.assertEqual(
            models,
            ["qwen2.5-coder:1.5b", "qwen2.5:3b", "qwen2.5:7b"],
        )

    def test_local_first_uses_ollama_when_model_is_available(self):
        with (
            mock.patch.object(llm_helper, "OLLAMA_PREFER_LOCAL", True),
            mock.patch.object(
                llm_helper, "ollama_model_is_installed", return_value=(True, "")
            ),
            mock.patch.object(
                llm_helper, "_call_ollama", return_value="local response"
            ) as ollama,
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=AssertionError("cloud called")
            ),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "local response")
        ollama.assert_called_once_with(
            "hello", llm_helper.OLLAMA_MODEL, False
        )

    def test_local_first_falls_back_to_cloud_when_ollama_unavailable(self):
        with (
            mock.patch.object(llm_helper, "OLLAMA_PREFER_LOCAL", True),
            mock.patch.object(
                llm_helper,
                "ollama_model_is_installed",
                return_value=(False, "Ollama API unavailable (ConnectionError)"),
            ),
            mock.patch.object(
                llm_helper, "_call_groq", return_value="cloud fallback"
            ) as groq,
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "cloud fallback")
        groq.assert_called_once_with(
            "hello", llm_helper.GROQ_MODEL, False
        )

    def test_router_kilo_empty_response(self):
        with (
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=_http_error(503)
            ),
            mock.patch.object(llm_helper, "_call_kilo", return_value=""),
            mock.patch.object(
                llm_helper, "_call_openrouter", return_value="hello from OpenRouter"
            ),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from OpenRouter")
        self.assertEqual(llm_helper._ROUTER_STATE["last_good_tier"], 3)

    def test_openrouter_empty_response_falls_back_to_groq(self):
        groq_calls = []

        def groq(prompt, model, json_mode):
            del prompt, json_mode
            groq_calls.append(model)
            if model == llm_helper.GROQ_MODEL:
                raise _http_error(503)
            return "hello from Groq fallback"

        with (
            mock.patch.object(llm_helper, "_call_groq", side_effect=groq) as groq_mock,
            mock.patch.object(llm_helper, "_call_kilo", side_effect=_http_error(503)),
            mock.patch.object(llm_helper, "_call_openrouter", return_value=""),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from Groq fallback")
        self.assertEqual(groq_mock.call_count, 2)
        self.assertEqual(groq_calls, [llm_helper.GROQ_MODEL, llm_helper.GROQ_FALLBACK_MODEL])
        self.assertEqual(llm_helper._ROUTER_STATE["last_good_tier"], 4)

    def test_openrouter_pool_fallback(self):
        calls = []

        def openrouter(prompt, model, json_mode):
            del prompt, json_mode
            calls.append(model)
            if model == "openrouter-primary":
                raise _http_error(429)
            if model == "openrouter-secondary":
                return "hello from second OpenRouter pool model"
            raise AssertionError(f"Unexpected OpenRouter model: {model}")

        groq = mock.Mock(side_effect=_http_error(503))
        with (
            mock.patch.object(
                llm_helper,
                "_call_kilo",
                side_effect=_http_error(503),
            ),
            mock.patch.object(llm_helper, "_call_openrouter", side_effect=openrouter),
            mock.patch.object(llm_helper, "_call_groq", groq),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from second OpenRouter pool model")
        self.assertEqual(calls, ["openrouter-primary", "openrouter-secondary"])
        self.assertEqual(groq.call_count, 1)
        self.assertEqual(
            llm_helper._ROUTER_STATE["last_good_model"],
            "openrouter-secondary",
        )

    def test_openrouter_skips_unlisted_model_in_pool(self):
        availability = mock.Mock(
            side_effect=[
                (False, "model not present in OpenRouter catalog"),
                (True, ""),
            ]
        )
        with (
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper,
                "OPENROUTER_FREE_POOL",
                (
                    "model_khong_ton_tai:free",
                    "nvidia/nemotron-3-super-120b-a12b:free",
                ),
            ),
            mock.patch.object(
                llm_helper,
                "OPENROUTER_PRIMARY_MODEL",
                "model_khong_ton_tai:free",
            ),
            mock.patch.object(
                llm_helper, "_call_kilo", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "openrouter_model_is_available", availability
            ),
            mock.patch.object(
                llm_helper, "_call_openrouter", return_value="hello from pool"
            ) as openrouter,
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from pool")
        availability.assert_has_calls([
            mock.call("model_khong_ton_tai:free"),
            mock.call("nvidia/nemotron-3-super-120b-a12b:free"),
        ])
        openrouter.assert_called_once_with(
            "hello", "nvidia/nemotron-3-super-120b-a12b:free", False
        )

    def test_empty_openrouter_pool_uses_legacy_alias(self):
        with (
            mock.patch.object(llm_helper, "OPENROUTER_FREE_POOL", ()),
            mock.patch.object(llm_helper, "OPENROUTER_MODEL", "legacy-model"),
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "_call_kilo", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "openrouter_model_is_available", return_value=(True, "")
            ),
            mock.patch.object(
                llm_helper, "_call_openrouter", return_value="hello from legacy"
            ) as openrouter,
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from legacy")
        openrouter.assert_called_once_with("hello", "legacy-model", False)

    def test_openrouter_catalog_is_cached(self):
        response = mock.Mock()
        response.json.return_value = {
            "data": [{"id": "catalog-model"}, {"id": "other-model"}]
        }
        response.raise_for_status.return_value = None
        with mock.patch.object(
            llm_helper.requests, "get", return_value=response
        ) as get:
            self.assertEqual(
                llm_helper._cached_openrouter_models()[0],
                {"catalog-model", "other-model"},
            )
            self.assertEqual(
                llm_helper._cached_openrouter_models()[0],
                {"catalog-model", "other-model"},
            )

        get.assert_called_once()

    def test_openrouter_blocks_model_after_three_recent_failures(self):
        with (
            mock.patch.object(llm_helper, "OPENROUTER_FREE_POOL", ("flaky-model",)),
            mock.patch.object(llm_helper, "OPENROUTER_PRIMARY_MODEL", "flaky-model"),
            mock.patch.object(
                llm_helper, "OPENROUTER_FALLBACK_MODEL", "fallback-model"
            ),
            mock.patch.object(llm_helper, "OPENROUTER_MODEL", "flaky-model"),
            mock.patch.object(
                llm_helper, "_call_kilo", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "_call_openrouter", side_effect=_http_error(429)
            ) as openrouter,
            mock.patch.object(
                llm_helper, "_call_groq", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "_call_gemini", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "_call_ollama", side_effect=_http_error(503)
            ),
            mock.patch.object(
                llm_helper, "ollama_model_is_installed", return_value=(True, "")
            ),
        ):
            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    llm_helper.call_llm("hello")

        self.assertEqual(openrouter.call_count, 6)
        self.assertGreater(
            llm_helper._ROUTER_STATE["blocked_models"]["flaky-model"],
            llm_helper.time.time(),
        )

    def test_groq_primary_falls_back(self):
        calls = []

        def groq(prompt, model, json_mode):
            del prompt, json_mode
            calls.append(model)
            if model == llm_helper.GROQ_MODEL:
                raise _http_error(429)
            return "hello from Groq fallback"

        with (
            mock.patch.object(
                llm_helper,
                "_call_kilo",
                side_effect=_http_error(503),
            ),
            mock.patch.object(
                llm_helper,
                "_call_openrouter",
                side_effect=_http_error(503),
            ),
            mock.patch.object(llm_helper, "_call_groq", side_effect=groq),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from Groq fallback")
        self.assertEqual(calls, [llm_helper.GROQ_MODEL, llm_helper.GROQ_FALLBACK_MODEL])

    def test_all_tiers_fail(self):
        fail = mock.Mock(side_effect=_http_error(429))
        output = io.StringIO()
        with (
            mock.patch.object(llm_helper, "_call_kilo", side_effect=fail),
            mock.patch.object(llm_helper, "_call_openrouter", side_effect=fail),
            mock.patch.object(llm_helper, "_call_gemini", side_effect=fail),
            mock.patch.object(llm_helper, "_call_groq", side_effect=fail),
            mock.patch.object(llm_helper, "_call_ollama", side_effect=fail),
            mock.patch.object(
                llm_helper,
                "ollama_model_is_installed",
                return_value=(True, ""),
            ),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaisesRegex(RuntimeError, "Tier 7"):
                llm_helper.call_llm("hello")

        logs = output.getvalue()
        for tier in range(1, 8):
            self.assertIn(f"Tier {tier}", logs)
        self.assertIn("Tier 6 (Gemini)", logs)
        self.assertIn("Tier 7 (Ollama)", logs)

    def test_router_all_fail_ollama(self):
        fail = mock.Mock(side_effect=_http_error(429))
        output = io.StringIO()
        with (
            mock.patch.object(llm_helper, "_call_kilo", side_effect=fail),
            mock.patch.object(llm_helper, "_call_openrouter", side_effect=fail),
            mock.patch.object(
                llm_helper, "OPENROUTER_PRIMARY_MODEL", "openrouter-primary"
            ),
            mock.patch.object(
                llm_helper, "OPENROUTER_FALLBACK_MODEL", "openrouter-fallback"
            ),
            mock.patch.object(llm_helper, "_call_groq", side_effect=fail),
            mock.patch.object(llm_helper, "_call_gemini", side_effect=fail),
            mock.patch.object(
                llm_helper,
                "ollama_model_is_installed",
                return_value=(True, ""),
            ),
            mock.patch.object(
                llm_helper, "_call_ollama", return_value="hello from local Ollama"
            ) as ollama,
            contextlib.redirect_stdout(output),
        ):
            result = llm_helper.call_llm("hello")

        self.assertEqual(result, "hello from local Ollama")
        ollama.assert_called_once_with(
            "hello", llm_helper.OLLAMA_MODEL, False
        )
        self.assertIn("Tier 7 (Ollama)", output.getvalue())
        self.assertEqual(llm_helper._ROUTER_STATE["last_good_tier"], 7)

    def test_explicit_ollama_model_does_not_index_cloud_tiers(self):
        with (
            mock.patch.object(llm_helper, "GROQ_API_KEY", ""),
            mock.patch.object(llm_helper, "KILO_API_KEY", ""),
            mock.patch.object(llm_helper, "OPENROUTER_API_KEY", ""),
            mock.patch.object(llm_helper, "GEMINI_API_KEY", ""),
            mock.patch.object(
                llm_helper, "ollama_model_is_installed", return_value=(True, "")
            ),
            mock.patch.object(
                llm_helper, "_call_ollama", return_value="local plan"
            ) as ollama,
        ):
            result = llm_helper.call_llm(
                "plan prompt", model="ollama:qwen2.5:3b", json_mode=False,
            )

        self.assertEqual(result, "local plan")
        ollama.assert_called_once_with("plan prompt", "qwen2.5:3b", False)

    def test_call_ollama_endpoint_and_timeout(self):
        response = mock.Mock()
        response.json.return_value = {"response": "hello"}
        with mock.patch.object(
            llm_helper.requests, "post", return_value=response
        ) as post:
            result = llm_helper._call_ollama(
                "hello", llm_helper.OLLAMA_MODEL, False
            )

        self.assertEqual(result, "hello")
        post.assert_called_once()
        self.assertEqual(
            post.call_args.args[0],
            f"{llm_helper.OLLAMA_HOST.rstrip('/')}/api/generate",
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 180)

    def test_state_persistence(self):
        order = []

        def groq(*_args):
            del _args
            order.append(1)
            raise _http_error(503)

        def kilo(*_args):
            del _args
            order.append(2)
            return "hello"

        def gemini(*_args):
            del _args
            order.append(6)
            raise AssertionError("Gemini should not run after Kilo succeeds")

        with (
            mock.patch.object(llm_helper, "_call_groq", side_effect=groq),
            mock.patch.object(llm_helper, "_call_kilo", side_effect=kilo),
            mock.patch.object(llm_helper, "_call_gemini", side_effect=gemini),
        ):
            self.assertEqual(llm_helper.call_llm("hello"), "hello")
            self.assertEqual(llm_helper.call_llm("hello"), "hello")

        self.assertEqual(order, [6, 1, 2, 6, 1, 2])
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["last_good_tier"], 2)
        self.assertEqual(state["daily_count"]["1"], 2)
        self.assertEqual(state["daily_count"]["2"], 2)
        self.assertEqual(state["last_reset_day"], llm_helper.date.today().isoformat())

    def test_tier_order(self):
        self.assertEqual(
            llm_helper.ROUTER_TIER_ORDER,
            (
                (1, "Groq", "qwen/qwen3.8-27b"),
                (2, "Kilo", "kilo-auto/free"),
                (3, "OpenRouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                (4, "Groq", "openai/gpt-oss-20b"),
                (5, "OpenRouter fallback", "nvidia/nemotron-3.5-lightning:free"),
                (6, "Gemini", "gemini-3.5-flash-lite"),
                (7, "Ollama", "qwen2.5-coder:1.5b"),
            ),
        )

    def test_explicit_swarm_route_falls_back_to_different_provider(self):
        groq = mock.Mock(side_effect=_http_error(429))
        openrouter = mock.Mock(side_effect=_http_error(429))
        kilo = mock.Mock(return_value="hello from Kilo")
        with (
            mock.patch.object(llm_helper, "_call_groq", groq),
            mock.patch.object(llm_helper, "_call_openrouter", openrouter),
            mock.patch.object(llm_helper, "_call_kilo", kilo),
            mock.patch.object(llm_helper, "_call_gemini", side_effect=_http_error(503)),
            mock.patch.object(llm_helper, "_call_ollama") as ollama,
        ):
            result = llm_helper.call_llm(
                "hello",
                model=f"groq:{llm_helper.GROQ_FALLBACK_MODEL}",
            )

        self.assertEqual(result, "hello from Kilo")
        self.assertEqual(groq.call_count, 1)
        self.assertEqual(openrouter.call_count, 1)
        kilo.assert_called_once()
        ollama.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
