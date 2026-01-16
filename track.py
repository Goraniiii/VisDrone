# track.py

import argparse
import yaml
from ultralytics import YOLO
import os
import json
from glob import glob

class DetectionTracker:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.model = YOLO(self.cfg["weights"])
        self.sequences_root = self.cfg["source"]

    def run(self):
        seq_names = sorted(os.listdir(self.sequences_root))
        for seq_name in seq_names:
            seq_path = os.path.join(self.sequences_root, seq_name)

            self.model.track(
                source=seq_path,
                imgsz=self.cfg["imgsz"],
                conf=self.cfg["conf"],
                iou=self.cfg["iou"],
                device=self.cfg["device"],
                tracker=self.cfg["tracker"],
                save=self.cfg["save"],
                save_txt=self.cfg["save_txt"],
                project=self.cfg["output_dir"],
                name=seq_name,
                exist_ok=True,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/track.yaml",
    )
    args = parser.parse_args()

    tracker = DetectionTracker(args.config)
    tracker.run()
