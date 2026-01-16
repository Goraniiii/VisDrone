# tracking_json.py

import os
import json
from glob import glob

def convert_sequence(seq_dir, output_json):
    label_dir = os.path.join(seq_dir, "labels")
    txt_files = sorted(glob(os.path.join(label_dir, "*.txt")))

    results = []

    for txt_path in txt_files:
        frame_id = int(os.path.basename(txt_path).split(".")[0])

        with open(txt_path, "r") as f:
            for line in f:
                cls, x, y, w, h, tid = line.strip().split()

                results.append({
                    "frame_id": frame_id,
                    "track_id": int(tid),
                    "category_id": int(cls),
                    "bbox": [
                        float(x),
                        float(y),
                        float(w),
                        float(h)
                    ],
                })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {output_json}")


if __name__ == "__main__":
    root = "runs/detect/results/tracking"

    for seq in os.listdir(root):
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue

        out_json = os.path.join(seq_dir, f"{seq}.json")
        convert_sequence(seq_dir, out_json)
