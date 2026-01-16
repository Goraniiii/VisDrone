# image_to_video.py

import os
import subprocess
from pathlib import Path


def images_to_video(img_dir, out_mp4, fps=30, start_number=1):
    img_dir = Path(img_dir)
    input_pattern = str(img_dir / f"%07d.jpg")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-framerate", str(fps),
        "-start_number", str(start_number),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd)


images_to_video(
    "runs/detect/results/tracking/uav0000086_00000_v",
    "runs/detect/results/tracking/uav0000086_00000_v.mp4"
)