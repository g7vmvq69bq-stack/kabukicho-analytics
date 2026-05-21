"""
Resolves a stream URL (YouTube, RTSP, HLS, or local file) into an
OpenCV VideoCapture object.  YouTube URLs are resolved via yt-dlp.
"""

import subprocess, json, sys, cv2


def _resolve_youtube(url: str) -> str:
    """Return the best direct video URL for a YouTube live stream."""
    cmd = [
        "yt-dlp",
        "--format", "best[height<=720][ext=mp4]/best[height<=720]/best",
        "--get-url",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed:\n{result.stderr.strip()}\n"
            "Make sure yt-dlp is installed: pip install yt-dlp"
        )
    direct_url = result.stdout.strip().split("\n")[0]
    return direct_url


def open_stream(url: str) -> cv2.VideoCapture:
    """
    Open any stream URL with OpenCV.
    Automatically resolves YouTube URLs through yt-dlp.
    """
    direct_url = url
    if "youtube.com" in url or "youtu.be" in url:
        print(f"[stream] Resolving YouTube URL via yt-dlp …")
        direct_url = _resolve_youtube(url)
        print(f"[stream] Direct URL obtained.")

    cap = cv2.VideoCapture(direct_url)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open stream: {direct_url}\n"
            "Check your internet connection or try a different stream URL."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[stream] Opened — {w}×{h} @ {fps:.1f} fps")
    return cap
