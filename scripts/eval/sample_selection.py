def parse_sample_indices(value, dataset_length):
    if value is None or not str(value).strip():
        return None
    indices = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("SAMPLE_INDICES must not contain duplicates")
    invalid = [index for index in indices if index < 0 or index >= dataset_length]
    if invalid:
        raise ValueError(
            f"SAMPLE_INDICES contains out-of-range values: {invalid}"
        )
    return indices
