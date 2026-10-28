"""Tests for the Perplexity Search API engine (``browser_use.integrations.perplexity``) on the
native ``search`` action.

Per the repo's testing rules these use real objects: the real ``httpx`` call in
``search_perplexity`` is pointed at a local ``pytest-httpserver`` via the
``PERPLEXITY_BASE_URL`` env var. No mocks, no real network.
"""

import json
import logging

import pytest
from pytest_httpserver import HTTPServer

from browser_use import BrowserProfile, BrowserSession
from browser_use.agent.views import ActionResult
from browser_use.integrations.perplexity import search_perplexity
from browser_use.integrations.perplexity.search import logger as perplexity_logger
from browser_use.tools.service import Tools


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


class _RecordingHandler(logging.Handler):
	"""Captures what the integration logger emits. browser_use loggers do not propagate to root,
	so pytest's caplog does not see them; attach directly instead."""

	def __init__(self) -> None:
		super().__init__(level=logging.DEBUG)
		self.lines: list[str] = []

	def emit(self, record: logging.LogRecord) -> None:
		self.lines.append(record.getMessage())

	@property
	def text(self) -> str:
		return '\n'.join(self.lines)


@pytest.fixture
def perplexity_log():
	handler = _RecordingHandler()
	previous_level = perplexity_logger.level
	perplexity_logger.addHandler(handler)
	perplexity_logger.setLevel(logging.DEBUG)
	yield handler
	perplexity_logger.removeHandler(handler)
	perplexity_logger.setLevel(previous_level)


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

	result = await search_perplexity('browser use release notes')

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

	result = await search_perplexity('fast query', max_results=3, search_type='fast')
	assert result.error is None
	request, _ = httpserver.log[-1]
	assert json.loads(request.get_data()) == {
		'query': 'fast query',
		'max_results': 3,
		'search_type': 'fast',
		'max_tokens_per_page': 256,
	}

	with pytest.raises(AssertionError):
		await search_perplexity('too many', max_results=21)
	with pytest.raises(AssertionError):
		await search_perplexity('bad type', search_type='people')


async def test_perplexity_search_handles_empty_results(httpserver, monkeypatch):
	"""An empty result set is reported without error."""
	httpserver.expect_request('/search', method='POST').respond_with_json({'id': 'x', 'results': []})
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('obscure query with no hits')

	assert result.error is None
	assert '0 results' in (result.extracted_content or '')


async def test_perplexity_search_http_error_returns_friendly_error(httpserver, monkeypatch):
	"""HTTP errors from the API become an ActionResult error with a status hint, not an exception."""
	httpserver.expect_request('/search', method='POST').respond_with_json({'detail': 'Unauthorized'}, status=401)
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('anything')

	assert result.extracted_content is None
	assert result.error is not None
	assert 'HTTP 401' in result.error
	assert 'PERPLEXITY_API_KEY' in result.error


async def test_perplexity_search_never_leaks_api_key(httpserver, monkeypatch, perplexity_log):
	"""A key with a trailing newline is stripped and used as-is; a key that is still an illegal
	header value makes httpx raise with the full Authorization header in the exception text,
	which must not reach ActionResult.error or the logs."""
	httpserver.expect_request('/search', method='POST').respond_with_json(_search_payload())
	monkeypatch.setenv('PERPLEXITY_BASE_URL', httpserver.url_for('/').rstrip('/'))

	# Trailing newline (common when the key is read from a file): stripped, request succeeds.
	monkeypatch.setenv('PERPLEXITY_API_KEY', 'pplx-secret-trailing\n')
	result = await search_perplexity('newline key')
	assert result.error is None
	request, _ = httpserver.log[-1]
	assert request.headers.get('Authorization') == 'Bearer pplx-secret-trailing'

	# Embedded newline cannot be fixed by strip(): httpx raises LocalProtocolError whose message
	# contains the raw header value. The error must be a fixed string with no key material.
	perplexity_log.lines.clear()
	monkeypatch.setenv('PERPLEXITY_API_KEY', 'pplx-secret\nembedded')
	result = await search_perplexity('second query')

	assert result.extracted_content is None
	assert result.error is not None
	assert 'LocalProtocolError' in result.error
	assert 'LocalProtocolError' in perplexity_log.text
	for leaked in ('pplx-secret', 'Bearer', 'embedded'):
		assert leaked not in result.error
		assert leaked not in perplexity_log.text


async def test_perplexity_search_does_not_echo_response_body(httpserver, monkeypatch, perplexity_log):
	"""Error response bodies are never copied into logs or ActionResult (they can echo request
	headers); only the status code and a fixed hint are reported."""
	sentinel = 'SENTINEL-DO-NOT-LOG-Bearer-test-key'
	httpserver.expect_request('/search', method='POST').respond_with_data(sentinel, status=500)
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('server error')

	assert result.extracted_content is None
	assert result.error == 'Failed to search Perplexity for "server error": HTTP 500 request failed'
	assert sentinel not in result.error
	assert sentinel not in perplexity_log.text
	assert 'Perplexity Search API returned HTTP 500' in perplexity_log.text


@pytest.mark.parametrize(
	('status', 'hint'),
	[
		(422, 'invalid request parameters'),
		(429, 'rate limited, retry later'),
	],
)
async def test_perplexity_search_422_and_429_use_fixed_hint(httpserver, monkeypatch, perplexity_log, status, hint):
	"""422 and 429 responses produce a fixed-hint error; the response body is never echoed."""
	body_sentinel = f'BODY-SENTINEL-{status}'
	httpserver.expect_request('/search', method='POST').respond_with_json(
		{'detail': [{'msg': body_sentinel, 'loc': ['body', 'max_results']}]}, status=status
	)
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('error query')

	assert result.extracted_content is None
	assert result.error is not None
	assert result.error == f'Failed to search Perplexity for "error query": HTTP {status} {hint}'
	assert body_sentinel not in result.error
	assert body_sentinel not in perplexity_log.text
	assert f'Perplexity Search API returned HTTP {status} ({hint})' in perplexity_log.text


async def test_perplexity_search_malformed_json_body(httpserver, monkeypatch, perplexity_log):
	"""A 200 with a non-JSON body takes the ValueError path: fixed message naming only the
	exception class, with none of the body or the exception text in the error or the logs."""
	body_sentinel = 'NOT-JSON-SENTINEL <html>oops</html>'
	httpserver.expect_request('/search', method='POST').respond_with_data(body_sentinel, status=200, content_type='text/html')
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('bad body')

	assert result.extracted_content is None
	assert result.error == 'Failed to search Perplexity for "bad body": JSONDecodeError'
	assert 'Perplexity Search API request failed: JSONDecodeError' in perplexity_log.text
	for leaked in (body_sentinel, 'Expecting value', 'line 1'):
		assert leaked not in result.error
		assert leaked not in perplexity_log.text


async def test_perplexity_search_malformed_base_url(monkeypatch, perplexity_log):
	"""httpx.InvalidURL is not an HTTPError. A malformed PERPLEXITY_BASE_URL must still produce the
	fixed class-name-only error, with no URL fragment in the error or the logs."""
	monkeypatch.setenv('PERPLEXITY_API_KEY', 'test-key')
	monkeypatch.setenv('PERPLEXITY_BASE_URL', 'http://[::1')

	result = await search_perplexity('bad base url')

	assert result.extracted_content is None
	assert result.error == 'Failed to search Perplexity for "bad base url": InvalidURL'
	assert 'Perplexity Search API request failed: InvalidURL' in perplexity_log.text
	for leaked in ('[::1', 'Invalid port', 'test-key'):
		assert leaked not in result.error
		assert leaked not in perplexity_log.text


@pytest.mark.parametrize(
	'payload',
	[
		['not', 'a', 'dict'],
		{'results': 'not a list'},
		{'results': {'title': 'dict instead of list'}},
		{'results': ['string item', {'title': 'ok', 'url': 'https://example.com', 'snippet': 'x'}]},
	],
)
async def test_perplexity_search_unexpected_response_shape(httpserver, monkeypatch, perplexity_log, payload):
	"""A JSON body that is not {results: [dict, ...]} produces a fixed error, not a traceback."""
	httpserver.expect_request('/search', method='POST').respond_with_json(payload)
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('odd shape')

	assert result.extracted_content is None
	assert result.error == 'Failed to search Perplexity for "odd shape": unexpected response shape'
	assert 'not a list' not in perplexity_log.text
	assert 'string item' not in perplexity_log.text


async def test_perplexity_search_missing_results_key_is_empty(httpserver, monkeypatch):
	"""A dict without a results key (or results: null) is treated as zero results, not an error."""
	httpserver.expect_request('/search', method='POST').respond_with_json({'id': 'x', 'results': None})
	_point_client_at(httpserver, monkeypatch)

	result = await search_perplexity('null results')

	assert result.error is None
	assert '0 results' in (result.extracted_content or '')
