# eval.py

import argparse
import json
import yaml
import random
import numpy as np
import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DetectionEvaluator:
    def __init__(self, config_path: str):
        self.cfg = self._load_config(config_path)
        self._set_seed(self.cfg.get("seed", 42))
        self.model = self._load_model()

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self):
        return YOLO(self.cfg["weights"])

    def evaluate(self):
        results = self.model.val(
            data=self.cfg["data"],
            imgsz=self.cfg["imgsz"],
            batch=self.cfg["batch"],
            device=self.cfg["device"],
            split="val"
        )

        metrics = {
            "dataset": "VisDrone-DET",
            "task": "object_detection",
            "model": self.cfg["model_name"],
            "epochs": self.cfg["epochs"],
            "imgsz": self.cfg["imgsz"],
            "batch": self.cfg["batch"],
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "seed": self.cfg["seed"],
        }

        self._save_metrics(metrics)
        return metrics

    def _save_metrics(self, metrics: dict):
        output_path = self.cfg["output"]
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Evaluation metrics saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisDrone Detection Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation config file",
    )
    args = parser.parse_args()

    evaluator = DetectionEvaluator(args.config)
    metrics = evaluator.evaluate()

    print("\nEvaluation Results")
    for k, v in metrics.items():
        print(f"{k}: {v}")
