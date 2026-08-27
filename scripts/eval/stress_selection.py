import argparse
import json
import re
from pathlib import Path


DEFAULT_STRESS_PREDICATES = {
    "support": ["on", "on top of", "on side of"],
    "containment": ["inside", "in"],
    "vertical": ["under", "below", "above"],
    "depth": ["behind", "in front of"],
    "interaction": ["holding", "riding", "wearing"],
}


def normalize_predicate(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def select_stress_indices(records, predicate_groups=None, per_group=8):
    predicate_groups = predicate_groups or DEFAULT_STRESS_PREDICATES
    normalized_groups = {
        group: {normalize_predicate(predicate) for predicate in predicates}
        for group, predicates in predicate_groups.items()
    }
    selected = {group: [] for group in normalized_groups}

    for record in records:
        predicates = {
            normalize_predicate(predicate)
            for predicate in record.get("predicates", [])
            if predicate
        }
        for group, targets in normalized_groups.items():
            if len(selected[group]) >= per_group:
                continue
            if predicates & targets:
                selected[group].append(int(record["index"]))
    return selected


def iter_h5_relation_records(h5_path, vocab_path):
    import h5py

    h5_path = Path(h5_path)
    vocab_path = Path(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    pred_idx_to_name = vocab["pred_idx_to_name"]

    with h5py.File(h5_path, "r") as h5:
        num_images = int(h5["image_ids"].shape[0])
        for index in range(num_images):
            num_relations = int(h5["relationships_per_image"][index])
            pred_indices = h5["relationship_predicates"][index][:num_relations]
            predicates = [
                str(pred_idx_to_name[int(predicate)]).lower()
                for predicate in pred_indices
                if int(predicate) < len(pred_idx_to_name)
            ]
            yield {
                "index": index,
                "image_id": int(h5["image_ids"][index]),
                "predicates": predicates,
            }


def flatten_selected_indices(selected):
    seen = set()
    indices = []
    for group_indices in selected.values():
        for index in group_indices:
            if index in seen:
                continue
            seen.add(index)
            indices.append(index)
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Select VG fixed-split stress samples by relation predicates."
    )
    parser.add_argument("--h5", required=True, help="Path to fixed-split h5.")
    parser.add_argument("--vocab", required=True, help="Path to fixed-split vocab.json.")
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    selected = select_stress_indices(
        iter_h5_relation_records(args.h5, args.vocab),
        per_group=args.per_group,
    )
    payload = {
        "groups": selected,
        "sample_indices": flatten_selected_indices(selected),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    print("SAMPLE_INDICES=" + ",".join(str(i) for i in payload["sample_indices"]))


if __name__ == "__main__":
    main()
