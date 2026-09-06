"""Keyword matching with deterministic semantic scores."""

import pytest

from oncorag.llm import prompt_builder


@pytest.fixture(autouse=True)
def zero_semantic_scores(monkeypatch):
    monkeypatch.setattr(
        prompt_builder.model_init,
        "get_combined_reranker_scores",
        lambda pairs, **kwargs: [0.0] * len(pairs),
    )


def rerank(context, query, **kwargs):
    return prompt_builder.rerank_context(
        context,
        query,
        top_k=1,
        runtime_options={
            "weights": {
                "semantic_weight": 0,
                "lexical_weight": 1,
                "name_weight": 0,
                "graph_weight": 0,
                "boost_alpha": 0,
                "penalty_beta": 0,
            },
            "graph_diffusion": {"enabled": False},
        },
        **kwargs,
    )


def test_configured_keyword_phrases_survive_concept_filtering():
    _, _, details = rerank(
        "Marker assay completed.\nUnrelated procedure completed.",
        "Marker result",
        normalized_name="Marker",
        keywords=["marker assay", "unrelated procedure"],
    )

    assert "marker assay" in details["keywords_used_for_boost"]
    assert "unrelated procedure" not in details["keywords_used_for_boost"]


@pytest.mark.parametrize(
    "marker,distractor",
    [("ER", "hereditary"), ("PR", "previous"), ("HER2", "HER20"), ("IDH", "IDH1")],
)
@pytest.mark.parametrize("source", ["name", "synonym", "compound_name"])
def test_configured_biomarkers_match_complete_tokens(marker, distractor, source):
    options = {
        "name": {"normalized_name": marker},
        "synonym": {"normalized_name": "Biomarker", "synonyms": [marker.lower()]},
        "compound_name": {"normalized_name": f"{marker} status"},
    }[source]
    positive = f"{marker.lower()}: positive."
    negative = f"Finding: {distractor}."

    selected, _, details = rerank(
        f"{negative}\n{positive}", "Result", **options,
    )
    by_sentence = {item["sentence"]: item for item in details["sentences_with_scores"]}

    assert selected == positive
    assert by_sentence[positive]["has_keyword"]
    assert by_sentence[positive]["lexical_score"] > 0
    assert not by_sentence[negative]["has_keyword"]
    assert by_sentence[negative]["lexical_score"] == 0


@pytest.mark.parametrize("label,distractor", [("yes", "yesterday"), ("no", "normal"), ("low", "lower")])
def test_short_categorical_query_terms_respect_word_boundaries(label, distractor):
    positive = f"Status: {label}."
    negative = f"Status: {distractor}."

    selected, _, details = rerank(
        f"{negative}\n{positive}",
        f"Is status {label}?",
        normalized_name="Status",
        expected_values=[label],
    )
    by_sentence = {item["sentence"]: item for item in details["sentences_with_scores"]}

    assert selected == positive
    assert by_sentence[positive]["lexical_score"] > by_sentence[negative]["lexical_score"]


def test_short_keyword_boundaries_apply_before_candidate_limit(monkeypatch):
    monkeypatch.setattr(
        prompt_builder,
        "_runtime_int",
        lambda key, env, default: 1 if key == "rerank_candidates" else default,
    )

    selected, _, details = rerank(
        "Previous report reviewed.\nPR: negative.",
        "Result",
        normalized_name="PR",
    )

    assert selected == "PR: negative."
    assert details["total_sentences_before"] == 1
