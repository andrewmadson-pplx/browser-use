"""
Perplexity integration for Browser Use.

Backs ``engine="perplexity"`` on the built-in ``search`` action with the Perplexity
Search API. Requires ``PERPLEXITY_API_KEY``.

Usage:
    from browser_use.integrations.perplexity import search_perplexity
    result = await search_perplexity('browser use release notes', max_results=5, search_type='fast')
"""

from .search import search_perplexity

__all__ = ['search_perplexity']
