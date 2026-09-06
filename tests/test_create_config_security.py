"""Ontology requests verify TLS and never expose credentials in error output."""

from unittest.mock import Mock

import pytest
import requests

from oncorag import create_config


SECRET = "test-api-secret-that-must-not-be-logged"


def response(data):
    result = Mock()
    result.json.return_value = data
    return result


def test_umls_search_uses_official_host_and_structured_parameters(monkeypatch):
    request = Mock(return_value=response({"result": {"results": [{"name": "Example", "ui": "C001"}]}}))
    monkeypatch.setattr(create_config.requests, "get", request)
    term = "TNF & IL-6 / receptor"
    assert create_config.search_umls(term, SECRET)[0]["cui"] == "C001"
    url = request.call_args.args[0]
    kwargs = request.call_args.kwargs
    assert url == "https://uts-ws.nlm.nih.gov/rest/search/current"
    assert SECRET not in url
    assert kwargs["params"] == {"string": term, "apiKey": SECRET}
    assert kwargs.get("verify", True) is True
    assert "Host" not in kwargs.get("headers", {})


def test_cui_request_encodes_path_and_keeps_credentials_out_of_url(monkeypatch):
    request = Mock(return_value=response({"result": {"name": "Example", "semanticTypes": []}}))
    monkeypatch.setattr(create_config.requests, "get", request)
    result = create_config.get_cui_semantic_types("C001?unexpected=yes", SECRET)
    assert result["name"] == "Example"
    url = request.call_args.args[0]
    assert url == "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C001%3Funexpected%3Dyes"
    assert SECRET not in url
    assert request.call_args.kwargs["params"] == {"apiKey": SECRET}
    assert request.call_args.kwargs.get("verify", True) is True


def test_bioportal_uses_https_and_documented_authorization_header(monkeypatch):
    request = Mock(return_value=response({"collection": []}))
    monkeypatch.setattr(create_config.requests, "get", request)
    assert create_config.search_bioportal("HER2 & ER", SECRET, limit=4) == []
    assert request.call_args.args[0] == "https://data.bioontology.org/search"
    assert request.call_args.kwargs["params"] == {"q": "HER2 & ER", "pagesize": 4, "suggest": "true"}
    assert request.call_args.kwargs["headers"] == {"Authorization": f"apikey token={SECRET}"}
    assert request.call_args.kwargs.get("verify", True) is True


@pytest.mark.parametrize("function", ["search_umls", "get_cui_semantic_types", "search_bioportal"])
def test_http_error_details_do_not_reach_logs_or_return_values(monkeypatch, capsys, function):
    failed = response({})
    failed.raise_for_status.side_effect = requests.HTTPError(f"401 for https://example.invalid/?apiKey={SECRET}")
    monkeypatch.setattr(create_config.requests, "get", Mock(return_value=failed))
    result = getattr(create_config, function)("C001", SECRET)
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err + repr(result)
    if isinstance(result, dict):
        assert result["error"] == "request_failed (HTTPError)"
    else:
        assert result == []


@pytest.mark.parametrize("function", ["search_umls", "get_cui_semantic_types", "search_bioportal"])
def test_timeout_retries_remain_bounded_without_logging_exception(monkeypatch, capsys, function):
    request = Mock(side_effect=requests.Timeout(f"Request URL contained apiKey={SECRET}"))
    pause = Mock()
    monkeypatch.setattr(create_config.requests, "get", request)
    monkeypatch.setattr(create_config.time, "sleep", pause)
    result = getattr(create_config, function)("C001", SECRET, max_retries=3)
    assert request.call_count == 3
    assert [call.args[0] for call in pause.call_args_list] == [1, 2]
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err + repr(result)
