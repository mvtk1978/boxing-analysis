#!/bin/bash
# Boxing Analysis Tool — Setup Script
set -e

echo "=== Boxing Video Analysis Tool Setup ==="
echo

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Try different install methods
install_packages() {
    PKGS="mediapipe opencv-python numpy yt-dlp scipy"

    echo "Installing packages: $PKGS"

    if pip3 install $PKGS --break-system-packages 2>/dev/null; then
        echo "✓ Installed with pip3 --break-system-packages"
        return 0
    fi

    if pip3 install --user $PKGS 2>/dev/null; then
        echo "✓ Installed with pip3 --user"
        return 0
    fi

    # Try pipx for yt-dlp at least
    if command -v pipx &>/dev/null; then
        pipx install yt-dlp 2>/dev/null || true
    fi

    echo "⚠ Manual install required:"
    echo "  pip3 install $PKGS --break-system-packages"
    return 1
}

install_packages

echo
echo "=== Setup complete ==="
echo
echo "Usage:"
echo "  Analyze a YouTube video:"
echo "    python3 analyze.py 'https://youtu.be/VIDEO_ID' --output result.mp4"
echo
echo "  Analyze with live preview:"
echo "    python3 analyze.py 'https://youtu.be/VIDEO_ID' --show --duration 30"
echo
echo "  Analyze only first 30 seconds:"
echo "    python3 analyze.py 'https://youtu.be/VIDEO_ID' --duration 30"
echo
echo "  Analyze local video:"
echo "    python3 analyze.py /path/to/video.mp4 --output result.mp4"
echo
echo "  Run synthetic demo (no video needed):"
echo "    python3 demo_synthetic.py"
