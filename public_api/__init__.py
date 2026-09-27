"""Read-only data service for the external ServingStudio site.

The Intro site's pages call this service through the site's own origin
(``/api/public/v1``); the site proxies those paths here. It is separate from
the Analyzer, which serves the internal UI. See ``public_api/README.md``.
"""
