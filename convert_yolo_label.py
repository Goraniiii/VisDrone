# convert_yolo_label.py

import os
import cv2
from tqdm import tqdm
import yaml

CONFIG_PATH = "configs/data_convert.yaml"

def convert_split(split):
    img_dir = os.path.join(ROOT, split, "images")
    ann_dir = os.path.join(ROOT, split, "annotations")
    label_dir = os.path.join(ROOT, split, "labels")
    os.makedirs(label_dir, exist_ok=True)

    ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".txt")]

    for ann_file in tqdm(ann_files, desc=f"Converting {split}"):
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, ann_file)
        label_path = os.path.join(label_dir, ann_file)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        yolo_lines = []

        with open(ann_path, "r") as f:
            for line in f:
                items = line.strip().split(",")
                if len(items) < 8:
                    continue

                x, y, bw, bh = map(float, items[:4])
                class_id = int(items[5])

                # filter
                if class_id not in VISDRONE_TO_YOLO:
                    continue

                # invalid
                if bw <= 0 or bh <= 0:
                    continue

                # to YOLO format
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bw /= w
                bh /= h

                cx = min(max(cx, 0), 1)
                cy = min(max(cy, 0), 1)
                bw = min(max(bw, 0), 1)
                bh = min(max(bh, 0), 1)

                yolo_class = VISDRONE_TO_YOLO[class_id]
                yolo_lines.append(
                    f"{yolo_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    ROOT = cfg["root"]
    SPLITS = cfg["splits"]
    VISDRONE_TO_YOLO = cfg["class_mapping"]

    for split in SPLITS:
        convert_split(split)
