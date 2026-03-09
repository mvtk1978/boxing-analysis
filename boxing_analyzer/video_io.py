"""
Video I/O helpers: download YouTube videos, open local files.
"""

import subprocess
import os
from pathlib import Path


def _get_video_codec(filepath: str) -> str:
    """Use ffprobe to detect video codec."""
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
    codec = _get_video_codec(filepath)
    if codec in ("av1", "vp9", "hevc", "h265"):
        print(f"[ffmpeg] Codec '{codec}' detected — re-encoding to H.264 for OpenCV compatibility...")
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


def download_youtube(url: str, output_dir: str = "/tmp/boxing_videos",
                     max_height: int = 480) -> str:
    """
    Download a YouTube video using yt-dlp.
    Prefers H.264 to avoid AV1 re-encoding step.
    Returns path to a H.264 compatible MP4 file.
    """
    os.makedirs(output_dir, exist_ok=True)
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

    print(f"[yt-dlp] Saved to: {filepath}")

    # Auto re-encode if needed
    filepath = ensure_h264(filepath)
    return filepath
