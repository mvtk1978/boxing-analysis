"""
Video I/O helpers: download YouTube videos, open local files.
"""

import subprocess
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse


_ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_safe_path(path: str, base_dir: str) -> bool:
    """Return True if path resolves within base_dir (prevents traversal)."""
    try:
        resolved = Path(path).resolve()
        base = Path(base_dir).resolve()
        return resolved == base or base in resolved.parents
    except Exception:
        return False


def is_youtube_url(url: str) -> bool:
    """Validate YouTube URL using proper URL parsing (not substring matching)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.netloc.lower().lstrip("www.")
        return host in ("youtube.com", "youtu.be", "m.youtube.com")
    except Exception:
        return False


def _get_video_codec(filepath: str) -> str:
    """Use ffprobe to detect video codec."""
    filepath = str(Path(filepath).resolve())
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
             filepath],
            capture_output=True, text=True
        )
        return result.stdout.strip().lower()
    except Exception:
        return "unknown"


def ensure_h264(filepath: str) -> str:
    """
    If video uses AV1 or other codecs that OpenCV can't decode,
    re-encode to H.264 using ffmpeg. Returns path to usable file.
    """
    filepath = str(Path(filepath).resolve())
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Video file not found: {filepath}")

    codec = _get_video_codec(filepath)
    if codec in ("av1", "vp9", "hevc", "h265"):
        print(f"[ffmpeg] Codec '{codec}' detected — re-encoding to H.264 for OpenCV compatibility...")
        # Place re-encoded file next to original, with fixed suffix
        out_path = filepath.rsplit(".", 1)[0] + "_h264.mp4"
        if os.path.exists(out_path):
            print(f"[ffmpeg] Re-encoded file exists: {out_path}")
            return out_path
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath,
             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
             "-c:a", "aac", out_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg re-encode failed:\n{result.stderr}")
        print(f"[ffmpeg] Re-encoded: {out_path}")
        return out_path
    return filepath


def download_youtube(url: str, output_dir: str = None,
                     max_height: int = 480) -> str:
    """
    Download a YouTube video using yt-dlp.
    Prefers H.264 to avoid AV1 re-encoding step.
    Returns path to a H.264 compatible MP4 file.
    """
    if not is_youtube_url(url):
        raise ValueError(f"URL does not appear to be a valid YouTube URL: {url}")

    if max_height < 1 or max_height > 4320:
        raise ValueError(f"max_height must be between 1 and 4320, got {max_height}")

    # Use a secure temp directory if none provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="boxing_dl_")
    else:
        # Resolve and validate caller-supplied path
        output_dir = str(Path(output_dir).resolve())

    os.makedirs(output_dir, mode=0o700, exist_ok=True)

    # Use a fixed safe template; yt-dlp sanitizes %(title)s but we cap it
    out_template = os.path.join(output_dir, "%(title).50s.%(ext)s")

    # Prefer H.264 (avc1) for direct OpenCV compatibility
    fmt = (
        f"bestvideo[height<={max_height}][vcodec^=avc1]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/best[height<={max_height}][ext=mp4]"
        f"/best[height<={max_height}]"
    )

    cmd = [
        "yt-dlp",
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",
        "--print", "after_move:filepath",
        url,
    ]

    print(f"[yt-dlp] Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    filepath = result.stdout.strip().split("\n")[-1].strip()
    if not os.path.exists(filepath):
        mp4s = sorted(Path(output_dir).glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if not mp4s:
            raise RuntimeError("No video file found after download")
        filepath = str(mp4s[0])

    # Verify the downloaded file stays within our output directory
    if not _is_safe_path(filepath, output_dir):
        raise RuntimeError(f"Downloaded file path escapes output directory: {filepath}")

    # Verify it has an allowed extension
    if Path(filepath).suffix.lower() not in _ALLOWED_VIDEO_EXTS:
        raise RuntimeError(f"Downloaded file has unexpected extension: {filepath}")

    print(f"[yt-dlp] Saved to: {filepath}")

    # Auto re-encode if needed
    filepath = ensure_h264(filepath)
    return filepath
