"""Tests for the Perplexity Search API engine on the native ``search`` action.

Per the repo's testing rules these use real objects: the real ``httpx`` call in
``_search_perplexity`` is pointed at a local ``pytest-httpserver`` via the
``PERPLEXITY_BASE_URL`` env var. No mocks, no real network.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from browser_use import BrowserProfile, BrowserSession
from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools, _search_perplexity


def _search_payload() -> dict:
	return {
		'id': '00000000-0000-0000-0000-000000000000',
		'results': [
			{
				'title': 'Example One',
				'url': 'https://example.com/one',
				'snippet': 'First result snippet.',
				'date': '2026-01-02',
				'last_updated': '2026-01-03',
			},
			{
				# date / last_updated are optional in the API response
				'title': 'Example Two',
				'url': 'https://example.com/two',
				'snippet': 'Second result\n\n  snippet with   whitespace.',
			},
		],
	}


def _point_client_at(httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Point the real httpx call at the local pytest-httpserver."""
	monkeypatch.setenv('PERPLEXITY_API_KEY', 'test-key')
	monkeypatch.setenv('PERPLEXITY_BASE_URL', httpserver.url_for('/').rstrip('/'))


# Real registry + real (headless) browser, mirroring tests/ci/test_search_find.py.


@pytest.fixture(scope='module')
async def browser_session():
	session = BrowserSession(browser_profile=BrowserProfile(headless=True, user_data_dir=None, keep_alive=True))
	await session.start()
	yield session
	await session.kill()


@pytest.fixture(scope='function')
def tools():
	return Tools()


# --- Through the real registered `search` action (engine routing + side-effects) ---


async def test_search_action_perplexity_engine_returns_results_inline(tools, browser_session, httpserver, monkeypatch):
	"""engine='perplexity' on the real search action returns structured results inline,
	without navigating the browser."""
	httpserver.expect_request('/search', method='POST').respond_with_json(_search_payload())
	_point_client_at(httpserver, monkeypatch)

	url_before = await browser_session.get_current_page_url()
	result = await tools.search(query='browser use release notes', engine='perplexity', browser_session=browser_session)

	assert isinstance(result, ActionResult)
	assert result.error is None
	assert result.include_extracted_content_only_once is True
	assert result.extracted_content is not None
	assert 'Example One' in result.extracted_content
	assert 'https://example.com/two' in result.extracted_content
	assert '2 results' in (result.long_term_memory or '')

	# The perplexity path returns before any NavigateToUrlEvent, so the browser does not move.
	assert await browser_session.get_current_page_url() == url_before

	# The request really reached the API with the expected body and headers.
	request, _ = httpserver.log[-1]
	assert request.headers.get('Authorization') == 'Bearer test-key'
	assert request.headers.get('X-Pplx-Integration') == 'browser-use'
	assert request.headers.get('User-Agent', '').startswith('browser-use/')
	body = json.loads(request.get_data())
	assert body == {'query': 'browser use release notes', 'max_results': 10, 'search_type': 'web', 'max_tokens_per_page': 256}


async def test_search_action_perplexity_missing_key(tools, browser_session, monkeypatch):
	"""Through the action: no key -> friendly error, no API call, no navigation."""
	monkeypatch.delenv('PERPLEXITY_API_KEY', raising=False)

	url_before = await browser_session.get_current_page_url()
	result = await tools.search(query='anything', engine='perplexity', browser_session=browser_session)

	assert result.error == 'PERPLEXITY_API_KEY environment variable not set'
	assert result.extracted_content is None
	assert await browser_session.get_current_page_url() == url_before


async def test_search_action_unknown_engine_errors(tools, browser_session):
	"""An unknown engine returns a clear ActionResult error, not a crash."""
	result = await tools.search(query='x', engine='altavista', browser_session=browser_session)

	assert isinstance(result, ActionResult)
	assert result.error is not None
	assert 'Unsupported search engine' in result.error
	assert 'perplexity' in result.error


# --- Helper-level coverage of the perplexity engine (no browser needed) ---


async def test_perplexity_search_returns_structured_results(httpserver, monkeypatch):
	"""The perplexity engine formats the API results into ActionResult.extracted_content."""
	httpserver.expect_request('/search', method='POST').respond_with_json(_search_payload())
	_point_client_at(httpserver, monkeypatch)

	result = await _search_perplexity('browser use release notes')

	assert result.error is None
	assert result.extracted_content is not None
	assert 'Example One (2026-01-02)' in result.extracted_content
	assert 'First result snippet.' in result.extracted_content
	# multi-line snippets are collapsed to one line so each result stays a 3-line block
	assert '- Example Two\n  https://example.com/two\n  Second result snippet with whitespace.' in result.extracted_content
	assert '2 results' in (result.long_term_memory or '')


async def test_perplexity_search_passes_parameters(httpserver, monkeypatch):
	"""max_results and search_type are forwarded to the API; out-of-range values are rejected."""
	httpserver.expect_request('/search', method='POST').respond_with_json(_search_payload())
	_point_client_at(httpserver, monkeypatch)

	result = await _search_perplexity('fast query', max_results=3, search_type='fast')
	assert result.error is None
	request, _ = httpserver.log[-1]
	assert json.loads(request.get_data()) == {
		'query': 'fast query',
		'max_results': 3,
		'search_type': 'fast',
		'max_tokens_per_page': 256,
	}

	with pytest.raises(AssertionError):
		await _search_perplexity('too many', max_results=21)
	with pytest.raises(AssertionError):
		await _search_perplexity('bad type', search_type='people')


async def test_perplexity_search_handles_empty_results(httpserver, monkeypatch):
	"""An empty result set is reported without error."""
	httpserver.expect_request('/search', method='POST').respond_with_json({'id': 'x', 'results': []})
	_point_client_at(httpserver, monkeypatch)

	result = await _search_perplexity('obscure query with no hits')

	assert result.error is None
	assert '0 results' in (result.extracted_content or '')


async def test_perplexity_search_http_error_returns_friendly_error(httpserver, monkeypatch):
	"""HTTP errors from the API become an ActionResult error with a status hint, not an exception."""
	httpserver.expect_request('/search', method='POST').respond_with_json({'detail': 'Unauthorized'}, status=401)
	_point_client_at(httpserver, monkeypatch)

	result = await _search_perplexity('anything')

	assert result.extracted_content is None
	assert result.error is not None
	assert 'HTTP 401' in result.error
	assert 'PERPLEXITY_API_KEY' in result.error
