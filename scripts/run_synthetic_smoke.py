"""Run actual models on all three small, paired multilingual synthetic example cohorts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oncorag.config.feature_schema import load_feature_specs
from oncorag.config.pipeline_config import load_pipeline_config
from oncorag.evaluation import evaluate_results, load_records
from oncorag.pipeline import run_pipeline, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="phi3:mini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/synthetic_smoke")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    summaries = {}
    for language in ("english", "german", "mixed"):
        config = load_pipeline_config(ROOT / f"configs/oncorag_synthetic_{language}.json")
        config["runtime"]["ollama"].update(host=args.ollama_host, model=args.ollama_model)
        config["outputs"]["root"] = str(output / language)
        config["features"]["generated_config_dir"] = str(output / language / "feature_configs")
        result = run_pipeline(config)
        evaluation = evaluate_results(
            result["results"], load_records(config["evaluation"]["gold_path"]),
            feature_specs=load_feature_specs(config["features"]["specifications"]),
            resamples=config["evaluation"]["bootstrap"]["resamples"],
            seed=config["runtime"]["random_seed"],
        )
        write_json(output / language / "evaluation.json", evaluation)
        summaries[language] = {
            "patients": result["patients"], "notes": result["notes"],
            "answers": evaluation["expected_predictions"],
            "exact_match": evaluation["exact_match"],
            "status_counts": evaluation["status_counts"],
            "run_fingerprint": result["run_fingerprint"],
        }
    summary = {"model": args.ollama_model, "results": summaries,
               "scope": "Three paired purpose-authored timelines, not held-out clinical validation"}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(row["exact_match"] == 1 for row in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
