from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_fixedsplit_ours_1000/fake"))
H5_PATH = Path(os.environ.get("H5_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5"))
VOCAB_PATH = Path(os.environ.get("VOCAB_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json"))
DETECT_THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.05"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.50"))
OUTPUT_JSON = os.environ.get("OUTPUT_JSON")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

VG_TO_COCO = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "guy": "person",
    "child": "person", "person": "person", "people": "person",
    "bike": "bicycle", "bicycle": "bicycle",
    "car": "car", "truck": "truck", "bus": "bus", "train": "train",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "airplane": "airplane", "plane": "airplane", "boat": "boat",
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse", "sheep": "sheep",
    "cow": "cow", "elephant": "elephant", "bear": "bear", "zebra": "zebra", "giraffe": "giraffe",
    "bench": "bench", "backpack": "backpack", "umbrella": "umbrella", "handbag": "handbag",
    "tie": "tie", "suitcase": "suitcase", "ball": "sports ball",
    "bottle": "bottle", "wine bottle": "bottle", "glass": "wine glass", "wine glass": "wine glass",
    "cup": "cup", "bowl": "bowl", "banana": "banana", "apple": "apple", "orange": "orange",
    "pizza": "pizza", "cake": "cake", "donut": "donut", "chair": "chair",
    "couch": "couch", "sofa": "couch", "plant": "potted plant", "potted plant": "potted plant",
    "bed": "bed", "table": "dining table", "dining table": "dining table",
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "laptop": "laptop", "mouse": "mouse", "keyboard": "keyboard",
    "cell phone": "cell phone", "phone": "cell phone", "book": "book", "clock": "clock",
    "vase": "vase", "teddy bear": "teddy bear", "refrigerator": "refrigerator",
    "fridge": "refrigerator", "sink": "sink", "oven": "oven", "microwave": "microwave",
    "toilet": "toilet",
}


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union


def compute_ap(recalls, precisions):
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class VGFixedSplitGT:
    def __init__(self, h5_path, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.object_idx_to_name = vocab["object_idx_to_name"]
        self.h5 = h5py.File(h5_path, "r")
        self.image_id_to_index = {
            int(image_id): idx for idx, image_id in enumerate(self.h5["image_ids"][:].tolist())
        }

    def canonical_gt(self, image_id):
        idx = self.image_id_to_index[int(image_id)]
        num_objects = int(self.h5["objects_per_image"][idx])
        names = self.h5["object_names"][idx][:num_objects].tolist()
        boxes = self.h5["object_boxes"][idx][:num_objects].tolist()
        out = []
        for name_idx, box in zip(names, boxes):
            vg_name = str(self.object_idx_to_name[int(name_idx)]).lower()
            coco_name = VG_TO_COCO.get(vg_name)
            if coco_name is None:
                continue
            out.append({"label": coco_name, "box": xywh_to_xyxy(box)})
        return out


class FasterRCNNDetector:
    def __init__(self, threshold):
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()
        self.threshold = threshold
        self.transform = transforms.ToTensor()

    @torch.no_grad()
    def detect(self, image):
        tensor = self.transform(image).to(DEVICE)
        outputs = self.model([tensor])[0]
        preds = []
        for score, label, box in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
            score_value = float(score.item())
            if score_value < self.threshold:
                continue
            coco_name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            if coco_name == "__background__" or coco_name == "N/A":
                continue
            preds.append(
                {
                    "label": coco_name,
                    "score": score_value,
                    "box": [float(v) for v in box.tolist()],
                }
            )
        return preds


def run():
    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Missing FAKE_DIR: {FAKE_DIR}")
    gt = VGFixedSplitGT(H5_PATH, VOCAB_PATH)
    detector = FasterRCNNDetector(DETECT_THRESHOLD)

    image_paths = sorted(FAKE_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No generated png files found in {FAKE_DIR}")

    gt_by_class = defaultdict(lambda: defaultdict(list))
    pred_by_class = defaultdict(list)

    for image_path in image_paths:
        image_id = int(image_path.stem)
        gt_items = gt.canonical_gt(image_id)
        for item in gt_items:
            gt_by_class[item["label"]][image_id].append({"box": item["box"], "matched": False})

        with Image.open(image_path).convert("RGB") as image:
            preds = detector.detect(image)
        for pred in preds:
            pred_by_class[pred["label"]].append(
                {
                    "image_id": image_id,
                    "score": pred["score"],
                    "box": pred["box"],
                }
            )

    ap_per_class = {}
    gt_instances_per_class = {}
    for label, image_map in gt_by_class.items():
        total_gt = sum(len(items) for items in image_map.values())
        if total_gt == 0:
            continue
        gt_instances_per_class[label] = total_gt
        preds = sorted(pred_by_class.get(label, []), key=lambda item: item["score"], reverse=True)
        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)

        for i, pred in enumerate(preds):
            candidates = image_map.get(pred["image_id"], [])
            best_iou = 0.0
            best_j = -1
            for j, gt_item in enumerate(candidates):
                if gt_item["matched"]:
                    continue
                iou = iou_xyxy(pred["box"], gt_item["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= IOU_THRESHOLD and best_j >= 0:
                tp[i] = 1.0
                candidates[best_j]["matched"] = True
            else:
                fp[i] = 1.0

        if len(preds) == 0:
            ap_per_class[label] = 0.0
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_per_class[label] = compute_ap(recalls, precisions)

    if not ap_per_class:
        raise RuntimeError("No canonical GT classes were available for AP computation.")

    map50 = float(np.mean(list(ap_per_class.values())))

    summary = {
        "protocol": "vg_fixed_split_detectable_canonical_subset",
        "metric": f"mAP@{IOU_THRESHOLD:.2f}",
        "fake_dir": str(FAKE_DIR),
        "h5_path": str(H5_PATH),
        "vocab_path": str(VOCAB_PATH),
        "detector_model": "torchvision::fasterrcnn_resnet50_fpn",
        "detector_threshold": DETECT_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "num_images": len(image_paths),
        "map50": map50,
        "ap_per_class": ap_per_class,
        "gt_instances_per_class": gt_instances_per_class,
    }

    print(
        "mAP@{:.2f}={:.4f} classes={} images={}".format(
            IOU_THRESHOLD,
            map50,
            len(ap_per_class),
            len(image_paths),
        ),
        flush=True,
    )

    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {output_path}", flush=True)


if __name__ == "__main__":
    run()
