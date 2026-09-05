import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "oncorag_full_pipeline.example.json"


def test_reviewer_requested_robustness_analyses_are_configured():
    config = json.loads(CONFIG_PATH.read_text())
    evaluation = config["evaluation"]

    assert evaluation["feature_complexity_stratification"]
    assert evaluation["confidence_calibration"]["labels"] == ["High", "Medium", "Low"]
    assert evaluation["top_k_ablation"]["values"] == [3, 5, 10]
    assert evaluation["retrieval_weight_sensitivity"]["relative_perturbations"] == [0.5, 1.5]
    assert evaluation["inter_rater_agreement"]["metric"] == "cohen_kappa"
    assert evaluation["model_comparison"]["context_windows"] == [4096, 131072]
    assert config["retrieval"]["graph_diffusion"]["iterations"] == 2
    assert config["temporal_anchoring"]["baseline"]["window_months"] == 9


def test_rare_class_sensitivity_matches_the_paper_protocol():
    rare_class = json.loads(CONFIG_PATH.read_text())["evaluation"]["rare_class_sensitivity"]

    assert rare_class == {"max_minority_count": 5, "max_prevalence": 0.1}
