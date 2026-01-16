# train.py

import argparse
import yaml
import random
import numpy as np
import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DetectionTrainer:
    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)
        self._set_seed(self.cfg.get("seed", 42))
        self.model = self._build_model()

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self) -> YOLO:
        return YOLO(self.cfg["model"])

    def train(self) -> None:
        self.model.train(
            data=self.cfg["data"],
            epochs=self.cfg["epochs"],
            imgsz=self.cfg["imgsz"],
            batch=self.cfg["batch"],
            optimizer=self.cfg["optimizer"],
            lr0=self.cfg["lr0"],
            weight_decay=self.cfg["weight_decay"],
            seed=self.cfg["seed"],
            device=self.cfg["device"],
            project=self.cfg["project"],
            name=self.cfg["name"],
            exist_ok=self.cfg.get("exist_ok", True),
        )

    def validate(self):
        metrics = self.model.val(
            data=self.cfg["data"],
            imgsz=self.cfg["imgsz"],
            batch=self.cfg["batch"],
            device=self.cfg["device"],
        )

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config file",
    )
    args = parser.parse_args()

    trainer = DetectionTrainer(args.config)

    trainer.train()
    trainer.validate()