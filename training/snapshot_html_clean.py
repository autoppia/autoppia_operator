"""Compress stored page HTML for fewer tokens in SFT and logs.

Strips non-interactive bloat (scripts, styles) and minifies inter-tag spacing.
Control with env ``AUTOPPIA_SNAPSHOT_HTML_COMPRESS`` (default: on).
"""

from __future__ import annotations

import os
import re

# 1 = strip scripts/styles, minify; 0 = store raw (still subject to char cap in harvester)
_COMPRESS = str(os.environ.get("AUTOPPIA_SNAPSHOT_HTML_COMPRESS", "1")).lower() in (
    "1",
    "true",
    "yes",
    "",
)


def _strip_scripts_styles_regex(html: str) -> str:
    out = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    out = re.sub(r"(?is)<style[^>]*>.*?</style>", "", out)
    out = re.sub(r"(?is)<!--.*?-->", "", out)
    return out


def _minify_space(html: str) -> str:
    out = re.sub(r">\s+", ">", html)
    out = re.sub(r"\s+<", "<", out)
    out = re.sub(r"[\r\n\t]+", " ", out)
    out = re.sub(r" {2,}", " ", out)
    return out.strip()


def clean_snapshot_html(html: str) -> str:
    """Return a more token-efficient serialisation of the same document tree.

    Prefer BeautifulSoup + lxml so markup stays valid. Falls back to regex if
    dependencies are missing or parsing fails.
    """
    if not html or not _COMPRESS:
        return html
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError:
        return _minify_space(_strip_scripts_styles_regex(html))
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return _minify_space(_strip_scripts_styles_regex(html))
    for tag in soup.find_all(["script", "style", "noscript", "iframe", "template"]):
        tag.decompose()
    for c in soup.find_all(string=True):
        if isinstance(c, Comment):
            c.extract()
    try:
        serialized = str(soup)
    except Exception:
        return _minify_space(_strip_scripts_styles_regex(html))
    return _minify_space(serialized)


__all__ = ["clean_snapshot_html"]
