from pathlib import Path


def test_standard_sg2im_eval_script_sets_network_turbo_and_hf_mirror():
    script = Path("run_standard_sg2im_ddim50_eval.sh").read_text(encoding="utf-8")

    assert "source /etc/network_turbo" in script
    assert "HF_ENDPOINT" in script


def test_fixedsplit_clean_config_uses_masked_graph_delta_geo_prediction():
    config = Path("configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml").read_text(
        encoding="utf-8"
    )

    assert "relation_geo_prediction_source: masked_graph_delta" in config


def test_fixedsplit_clean_full_eval_script_uses_clean_protocol_defaults():
    script = Path("run_fixedsplit_geopred_clean_ddim50_eval_clean.sh").read_text(
        encoding="utf-8"
    )

    assert "source /etc/network_turbo" in script
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in script
    assert 'MODEL_YAML="${MODEL_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"' in script
    assert 'DATA_YAML="${DATA_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"' in script
    assert 'H5_PATH="${H5_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5}"' in script
    assert 'VOCAB_PATH="${VOCAB_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json}"' in script
    assert 'IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/fixed_split_work/datasets/vg/images}"' in script
    assert 'NUM_SAMPLES="${NUM_SAMPLES:-5096}"' in script
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"' in script
    assert 'SEED="${SEED:-20260508}"' in script


def test_vg_eval_generator_writes_optional_sample_metadata():
    script = Path("scripts/eval/generate_vg_fixedsplit_eval.py").read_text(
        encoding="utf-8"
    )

    assert "SAVE_SAMPLE_METADATA" in script
    assert "write_sample_metadata" in script
    assert "sample_metadata" in script


def test_relation_token_config_and_eval_script_enable_union_box_tokens():
    config = Path("configs/vg_standard_sg2im_scene_graph_relation_tokens.yaml").read_text(
        encoding="utf-8"
    )
    script = Path("run_standard_sg2im_relation_tokens_stress_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "enable_relation_grounding_tokens: true" in config
    assert "max_relation_grounding_tokens: 5" in config
    assert "ENABLE_RELATION_GROUNDING_TOKENS" in script
    assert "MAX_RELATION_GROUNDING_TOKENS" in script
    assert "SAMPLE_INDICES" in script
    assert 'RELATION_GROUNDING_TEMPLATE:-{subject}' not in script
    assert "export RELATION_GROUNDING_TEMPLATE='{subject} {predicate} {object}'" in script


def test_relation_token_v2_config_and_eval_script_filter_and_dedup():
    config = Path("configs/vg_standard_sg2im_scene_graph_relation_tokens_v2.yaml").read_text(
        encoding="utf-8"
    )
    script = Path("run_standard_sg2im_relation_tokens_v2_stress_eval.sh").read_text(
        encoding="utf-8"
    )

    assert "max_relation_grounding_tokens: 3" in config
    assert "deduplicate_relation_grounding_tokens: true" in config
    assert "relation_grounding_allowed_predicates:" in config
    assert '"holding"' in config
    assert '"on top of"' in config
    assert "MAX_RELATION_GROUNDING_TOKENS=\"${MAX_RELATION_GROUNDING_TOKENS:-3}\"" in script
    assert "DEDUP_RELATION_GROUNDING_TOKENS=\"${DEDUP_RELATION_GROUNDING_TOKENS:-1}\"" in script
    assert "RELATION_GROUNDING_ALLOWED_PREDICATES" in script
    assert "spatial relation: {subject} {predicate} {object}" in script
