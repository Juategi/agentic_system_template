# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - IMAGE UTILITIES
# =============================================================================
"""
Image Utilities Module

Extracts, downloads, and encodes images from GitHub issue markdown.
Used to provide visual context (mockups, screenshots, diagrams) to
agents that support multimodal LLM calls.

Features:
    - Extract image URLs from markdown syntax and bare URLs
    - Download images with authenticated GitHub sessions
    - Base64 encode for LLM API consumption
    - Configurable limits (max images, max size)
    - Graceful per-image error handling

Usage:
    from agents.base.image_utils import process_issue_images

    images = process_issue_images(
        issue_body,
        session=github_helper.session,
        max_images=5,
    )
    # images is a List[ImageContent] ready for LLM calls
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ImageContent:
    """
    A processed image ready for LLM consumption.

    Attributes:
        url: Original image URL.
        alt_text: Alt text from markdown (may be empty).
        media_type: MIME type (e.g. "image/png").
        base64_data: Base64-encoded image data.
        source: Where the image was found ("issue_body", "comment").
        size_bytes: Original image size in bytes.
    """
    url: str
    alt_text: str
    media_type: str
    base64_data: str
    source: str = "issue_body"
    size_bytes: int = 0


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_MEDIA_TYPES: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Content-Type header values mapped to canonical media types
_CONTENT_TYPE_MAP: Dict[str, str] = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}

# GitHub hosts that serve uploaded images
_GITHUB_IMAGE_HOSTS = frozenset([
    "user-images.githubusercontent.com",
    "github.com",
    "raw.githubusercontent.com",
    "private-user-images.githubusercontent.com",
])

# Image file extension pattern
_IMAGE_EXT_PATTERN = r"\.(?:png|jpe?g|gif|webp)"


# =============================================================================
# IMAGE URL EXTRACTION
# =============================================================================


def extract_image_urls(markdown_text: str) -> List[Dict[str, str]]:
    """
    Extract image URLs from markdown text.

    Detects:
    - ``![alt text](url)`` markdown image syntax
    - Bare image URLs ending in image extensions
    - GitHub user-content image URLs (may lack file extension)

    Args:
        markdown_text: Raw markdown string (e.g. GitHub issue body).

    Returns:
        Deduplicated list of ``{"url": ..., "alt_text": ...}`` dicts.
    """
    if not markdown_text:
        return []

    found: Dict[str, str] = {}  # url -> alt_text (dedup by URL)

    # Pattern 1: Markdown image syntax  ![alt](url)
    md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    for match in re.finditer(md_pattern, markdown_text):
        alt_text = match.group(1).strip()
        url = match.group(2).strip()
        if _is_image_url(url):
            found.setdefault(url, alt_text)

    # Pattern 2: Bare image URLs (not already inside markdown image syntax)
    # Match URLs that end with an image extension
    bare_pattern = (
        r"(?<!\()(?<!\])\b(https?://[^\s<>\"')\]]+?" + _IMAGE_EXT_PATTERN + r")(?:\?[^\s<>\"')\]]*)?(?=[\s<>\"')\],$]|$)"
    )
    for match in re.finditer(bare_pattern, markdown_text, re.IGNORECASE):
        url = match.group(0).strip()
        found.setdefault(url, "")

    # Pattern 3: GitHub user-content URLs (may not have file extension)
    gh_pattern = r"(https?://(?:user-images|private-user-images)\.githubusercontent\.com/[^\s<>\"')\]]+)"
    for match in re.finditer(gh_pattern, markdown_text):
        url = match.group(1).strip()
        found.setdefault(url, "")

    return [{"url": url, "alt_text": alt} for url, alt in found.items()]


def _is_image_url(url: str) -> bool:
    """Check if a URL points to an image based on extension or host."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    # Check file extension
    for ext in SUPPORTED_MEDIA_TYPES:
        if path_lower.endswith(ext):
            return True

    # Check known GitHub image hosts
    if parsed.hostname in _GITHUB_IMAGE_HOSTS:
        return True

    return False


# =============================================================================
# IMAGE DOWNLOAD
# =============================================================================


def download_image(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    max_size_bytes: int = 5_000_000,
) -> Tuple[bytes, str]:
    """
    Download a single image from a URL.

    Args:
        url: Image URL to download.
        session: Requests session (may include auth headers for private repos).
        timeout: Download timeout in seconds.
        max_size_bytes: Maximum image size to accept.

    Returns:
        Tuple of (raw_bytes, media_type).

    Raises:
        ValueError: If content is not an image or exceeds size limit.
        requests.RequestException: On network errors.
    """
    requester = session or requests
    response = requester.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Determine media type from Content-Type header
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    media_type = _CONTENT_TYPE_MAP.get(content_type)

    # Fallback: infer from URL extension
    if not media_type:
        media_type = _media_type_from_url(url)

    if not media_type:
        raise ValueError(
            f"Unsupported content type '{content_type}' from {url}"
        )

    # Read content with size limit
    chunks = []
    total_size = 0
    for chunk in response.iter_content(chunk_size=65536):
        total_size += len(chunk)
        if total_size > max_size_bytes:
            raise ValueError(
                f"Image exceeds size limit ({total_size} > {max_size_bytes} bytes): {url}"
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    if not raw_bytes:
        raise ValueError(f"Empty image response from {url}")

    return raw_bytes, media_type


def _media_type_from_url(url: str) -> Optional[str]:
    """Infer media type from URL file extension."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    for ext, mtype in SUPPORTED_MEDIA_TYPES.items():
        if path_lower.endswith(ext):
            return mtype
    return None


# =============================================================================
# BASE64 ENCODING
# =============================================================================


def encode_image_to_base64(raw_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string."""
    return base64.b64encode(raw_bytes).decode("utf-8")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def process_issue_images(
    markdown_text: str,
    session: Optional[requests.Session] = None,
    max_images: int = 5,
    max_size_bytes: int = 5_000_000,
    source: str = "issue_body",
) -> List[ImageContent]:
    """
    Extract, download, and encode images from issue markdown.

    This is the main entry point for the image processing pipeline.
    Each image is processed independently — a failure on one image
    does not block the others.

    Args:
        markdown_text: GitHub issue body or comment markdown.
        session: Authenticated requests.Session for private repos.
        max_images: Maximum number of images to process.
        max_size_bytes: Maximum size per image in bytes.
        source: Label for where images come from ("issue_body", "comment").

    Returns:
        List of successfully processed ``ImageContent`` objects.
    """
    url_entries = extract_image_urls(markdown_text)

    if not url_entries:
        return []

    if len(url_entries) > max_images:
        logger.info(
            f"Found {len(url_entries)} images, limiting to {max_images}"
        )
        url_entries = url_entries[:max_images]

    images: List[ImageContent] = []

    for entry in url_entries:
        url = entry["url"]
        alt_text = entry["alt_text"]

        try:
            raw_bytes, media_type = download_image(
                url,
                session=session,
                max_size_bytes=max_size_bytes,
            )

            base64_data = encode_image_to_base64(raw_bytes)

            images.append(ImageContent(
                url=url,
                alt_text=alt_text,
                media_type=media_type,
                base64_data=base64_data,
                source=source,
                size_bytes=len(raw_bytes),
            ))

            logger.debug(
                f"Processed image: {url} ({media_type}, {len(raw_bytes)} bytes)"
            )

        except ValueError as e:
            logger.warning(f"Skipping image {url}: {e}")
        except requests.RequestException as e:
            logger.warning(f"Failed to download image {url}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error processing image {url}: {e}")

    logger.info(
        f"Processed {len(images)}/{len(url_entries)} images from {source}"
    )

    return images


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ImageContent",
    "SUPPORTED_MEDIA_TYPES",
    "extract_image_urls",
    "download_image",
    "encode_image_to_base64",
    "process_issue_images",
]