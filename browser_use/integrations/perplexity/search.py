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

	``max_tokens_per_page`` caps the snippet length per result. The API default is
	much larger than an agent step needs, so a smaller value is sent by default.
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
	except Exception as e:
		# httpx.HTTPError (transport, timeout, protocol), httpx.InvalidURL (bad PERPLEXITY_BASE_URL,
		# not an HTTPError subclass), ValueError (non-JSON body), and anything else: report the class
		# name only, so no URL fragment or header value can reach the agent via the registry.
		logger.error(f'Perplexity Search API request failed: {type(e).__name__}')
		return ActionResult(error=f'Failed to search Perplexity for "{query}": {type(e).__name__}')

	results = data.get('results') if isinstance(data, dict) else None
	if isinstance(data, dict) and results is None:
		results = []
	if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
		logger.error('Perplexity Search API returned an unexpected response shape')
		return ActionResult(error=f'Failed to search Perplexity for "{query}": unexpected response shape')

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
