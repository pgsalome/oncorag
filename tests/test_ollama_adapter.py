"""Legacy Ollama retries must preserve the configured endpoint and generation."""

from unittest.mock import Mock

import pytest

from oncoraggraph.config.system_config import LLMBackend, LLMConfig
from oncoraggraph.llm.llm_adapter import OllamaAdapter


def configuration(**changes):
    return LLMConfig(**{
        "backend": LLMBackend.OLLAMA_LOCAL, "model_name": "configured-model",
        "requires_phi_removal": False, "base_url": "http://127.0.0.1:11435",
        "timeout": 23, "temperature": .1, "num_ctx": 4096, "max_tokens": 512,
        "extra_params": {"seed": 17, "top_p": .9}, **changes,
    })


def test_configured_client_and_options_apply_to_initial_and_retry_calls(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "environment-model")
    client = Mock()
    client.chat.side_effect = [
        {"message": {"content": "invalid json"}},
        {"message": {"content": '{"value": 52, "confidence": "High"}'}},
    ]
    constructor = Mock(return_value=client)
    monkeypatch.setattr("oncoraggraph.llm.llm_adapter.ollama.Client", constructor)
    monkeypatch.setattr(OllamaAdapter, "_prepare_context", lambda self, value: value)
    adapter = OllamaAdapter(configuration())
    assert adapter.query("Extract age from clinical evidence", "clinical evidence")["value"] == 52
    constructor.assert_called_once_with(host="http://127.0.0.1:11435", timeout=23)
    assert client.chat.call_count == 2
    for call in client.chat.call_args_list:
        assert call.kwargs["model"] == "environment-model"
        assert call.kwargs["options"] == {"seed": 17, "top_p": .9, "temperature": .1, "num_ctx": 4096, "num_predict": 512}
        assert call.kwargs["format"] == "json"
    assert client.chat.call_args.kwargs["messages"][0]["content"] == "Extract age from clinical evidence"


def test_host_environment_override_and_optional_limits(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11436")
    constructor = Mock()
    monkeypatch.setattr("oncoraggraph.llm.llm_adapter.ollama.Client", constructor)
    adapter = OllamaAdapter(configuration(num_ctx=None, max_tokens=None))
    constructor.assert_called_once_with(host="http://127.0.0.1:11436", timeout=23)
    assert "num_ctx" not in adapter._generation_options()
    assert "num_predict" not in adapter._generation_options()


@pytest.mark.parametrize("invalid", ["[]", "null", '{"value": NaN}', "malformed"])
def test_retries_are_bounded_and_reject_nonobject_or_nonfinite_json(monkeypatch, invalid):
    monkeypatch.setenv("ONCORAGGRAPH_LLM_JSON_MAX_TRIES", "3")
    client = Mock()
    client.chat.return_value = {"message": {"content": invalid}}
    monkeypatch.setattr("oncoraggraph.llm.llm_adapter.ollama.Client", Mock(return_value=client))
    monkeypatch.setattr(OllamaAdapter, "_prepare_context", lambda self, value: value)
    result = OllamaAdapter(configuration()).query("prompt", "context")
    assert result["value"] == "Missing"
    assert result["confidence"] == "Low"
    assert "error_during_extraction" in result["reasoning"]
    assert client.chat.call_count == 3
