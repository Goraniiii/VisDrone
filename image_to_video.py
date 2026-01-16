# image_to_video.py

import ffmpeg
from pathlib import Path


def make_video(
    image_dir: str,
    output_path: str,
    fps: int = 30,
    start_number: int = 1,
    img_ext: str = "jpg"
):
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    assert image_dir.exists(), f"{image_dir} not found"

    input_pattern = str(image_dir / f"%06d.{img_ext}")

    (
        ffmpeg
        .input(input_pattern, framerate=fps, start_number=start_number)
        .output(
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            movflags="+faststart"
        )
        .overwrite_output()
        .run(quiet=False)
    )


if __name__ == "__main__":
    make_video(
        image_dir="runs/detect/results/tracking/uav0000086_00000_v",
        output_path="runs/detect/results/tracking/uav0000086_00000_v.mp4",
        fps=30
    )
