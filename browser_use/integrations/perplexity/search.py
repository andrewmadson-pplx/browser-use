"""Perplexity Search API backend for the built-in ``search`` action (``engine="perplexity"``)."""

import logging
import os

import httpx

from browser_use.agent.views import ActionResult
from browser_use.utils import get_browser_use_version

logger = logging.getLogger(__name__)


async def search_perplexity(
	query: str, *, max_results: int = 10, search_type: str = 'web', max_tokens_per_page: int = 256
) -> ActionResult:
	"""Search the web via the Perplexity Search API.

	Unlike the browser-navigation engines (google/bing/duckduckgo), this returns
	structured results (title, url, snippet) directly in a single API call, with
	no page navigation. Requires a ``PERPLEXITY_API_KEY`` environment variable.
	``PERPLEXITY_BASE_URL`` overrides the API host (used by tests and proxies).

	``max_tokens_per_page`` caps the snippet length per result; the API default
	(~1024 tokens per result) would put ~10k tokens into the agent context per
	search, so we default to a much smaller value here.
	"""
	assert 1 <= max_results <= 20, 'max_results must be between 1 and 20'
	assert search_type in ('web', 'fast'), "search_type must be 'web' or 'fast'"
	assert max_tokens_per_page > 0, 'max_tokens_per_page must be positive'

	api_key = os.environ.get('PERPLEXITY_API_KEY', '').strip()
	if not api_key:
		return ActionResult(error='PERPLEXITY_API_KEY environment variable not set')

	base_url = os.environ.get('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai').rstrip('/')
	headers = {
		'Authorization': f'Bearer {api_key}',
		'Content-Type': 'application/json',
		'User-Agent': f'browser-use/{get_browser_use_version()}',
		'X-Pplx-Integration': 'browser-use',
	}
	payload = {
		'query': query,
		'max_results': max_results,
		'search_type': search_type,
		'max_tokens_per_page': max_tokens_per_page,
	}

	# Error messages below are deliberately fixed strings. Exception text is never
	# interpolated: httpx errors (e.g. LocalProtocolError on a malformed header) and
	# response bodies can echo request headers, including the Authorization value,
	# and ActionResult.error ends up in agent history and logs.
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(f'{base_url}/search', json=payload, headers=headers)
			response.raise_for_status()
			data = response.json()
	except httpx.HTTPStatusError as e:
		status = e.response.status_code
		hint = {401: 'check PERPLEXITY_API_KEY', 422: 'invalid request parameters', 429: 'rate limited, retry later'}.get(
			status, 'request failed'
		)
		logger.error(f'Perplexity Search API returned HTTP {status} ({hint})')
		return ActionResult(error=f'Failed to search Perplexity for "{query}": HTTP {status} {hint}')
	except (httpx.HTTPError, ValueError) as e:
		# HTTPError covers transport, timeout, and protocol errors; ValueError covers a non-JSON body.
		logger.error(f'Perplexity Search API request failed: {type(e).__name__}')
		return ActionResult(error=f'Failed to search Perplexity for "{query}": {type(e).__name__}')

	results = data.get('results') or []
	memory = f"Searched Perplexity for '{query}' ({len(results)} results)"
	if not results:
		return ActionResult(extracted_content=memory, long_term_memory=memory)

	lines = []
	for result in results:
		title = result.get('title') or ''
		url = result.get('url') or ''
		snippet = ' '.join((result.get('snippet') or '').split())
		date = result.get('date')
		header = f'- {title}' + (f' ({date})' if date else '')
		lines.append(f'{header}\n  {url}\n  {snippet}'.rstrip())
	extracted = f"Search results for '{query}':\n" + '\n'.join(lines)
	logger.info(f'🔍  {memory}')
	return ActionResult(
		extracted_content=extracted,
		long_term_memory=memory,
		include_extracted_content_only_once=True,
	)
