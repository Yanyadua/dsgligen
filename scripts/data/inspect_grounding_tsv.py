import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.tsv import TSVFile
from dataset.tsv_dataset import decode_item


def spans_to_phrase(caption, spans):
    parts = []
    for span in spans or []:
        if len(span) != 2:
            continue
        start, end = span
        parts.append(caption[start:end].strip())
    return " ".join(part for part in parts if part).strip()


def load_relation_ids(path):
    """Load image/data ids from common VG/GQA-style JSON relation files."""
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = set()
    records = data.values() if isinstance(data, dict) else data
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in ("image_id", "data_id", "id"):
            if key in record:
                ids.add(str(record[key]))
                break
    return ids


def main():
    parser = argparse.ArgumentParser(description="Inspect GLIGEN grounding TSV structure and optional SG id coverage.")
    parser.add_argument("--tsv", default="DATA/GROUNDING/gqa/tsv/train-00.tsv")
    parser.add_argument("--relations_json", default=None, help="Optional VG/GQA scene graph JSON for id coverage checks.")
    parser.add_argument("--samples", default="0,1,2,8,9,18,189")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    tsv = TSVFile(str(tsv_path))
    total = len(tsv) if args.max_rows is None else min(len(tsv), args.max_rows)
    relation_ids = load_relation_ids(args.relations_json)

    anno_counts = []
    phrase_counts = Counter()
    matched_relation_ids = 0
    for idx in range(total):
        key, raw = tsv[idx]
        item = decode_item(raw)
        caption = item.get("caption", "")
        annos = item.get("annos", [])
        anno_counts.append(len(annos))
        if relation_ids is not None and str(item.get("data_id", key)) in relation_ids:
            matched_relation_ids += 1
        for anno in annos:
            phrase = spans_to_phrase(caption, anno.get("tokens_positive"))
            if phrase:
                phrase_counts[phrase.lower()] += 1

    anno_tensor = torch.tensor(anno_counts, dtype=torch.float32)
    print("tsv:", tsv_path)
    print("rows:", len(tsv), "scanned:", total)
    print("annos_per_image: mean={:.2f} min={} max={}".format(
        anno_tensor.mean().item(), int(anno_tensor.min().item()), int(anno_tensor.max().item())
    ))
    if relation_ids is None:
        print("relations_json: not provided")
    else:
        print("relations_json_ids:", len(relation_ids))
        print("matched_rows:", matched_relation_ids, "coverage={:.2f}%".format(100 * matched_relation_ids / max(total, 1)))

    print("\ntop_phrases:")
    for phrase, count in phrase_counts.most_common(30):
        print(f"  {phrase}: {count}")

    print("\nsamples:")
    for idx_text in args.samples.split(","):
        idx = int(idx_text.strip())
        key, raw = tsv[idx]
        item = decode_item(raw)
        caption = item.get("caption", "")
        print("\nidx:", idx, "key:", key, "data_id:", item.get("data_id"))
        print("caption:", caption)
        for anno in item.get("annos", [])[:20]:
            phrase = spans_to_phrase(caption, anno.get("tokens_positive"))
            print("  anno_id:", anno.get("id"), "bbox:", anno.get("bbox"), "phrase:", phrase)


if __name__ == "__main__":
    main()
