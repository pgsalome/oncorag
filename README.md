# OncoRAG

[![Paper](https://img.shields.io/badge/npj%20Digital%20Medicine-10.1038%2Fs41746--026--03170--8-blue)](https://www.nature.com/articles/s41746-026-03170-8)
[![License](https://img.shields.io/badge/use-noncommercial_only-4B5563)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)

**Extract structured, typed clinical features from patient-specific knowledge
graphs with traceable source evidence.**

Define your variables, import dated and classified notes, and build one graph
per patient.

An optional patient chatbot uses the same graphs and retrieval configuration.

Paper: [OncoRAG in npj Digital Medicine](https://www.nature.com/articles/s41746-026-03170-8),
DOI `10.1038/s41746-026-03170-8`.

[![OncoRAG study workflow: clinical notes, configuration, extraction, retrieval, generation, and downstream prediction](graphicalabstract.png)](graphicalabstract.png)

*Study workflow overview. The progression-free-survival analysis is the paper's
downstream evaluation, not a built-in clinical prediction tool.*

## Table Of Contents

1. [Quick Start](#quick-start)
2. [Your Variables](#your-variables)
3. [Your Notes](#your-notes)
4. [Parameters And Outputs](#parameters-and-outputs)
5. [ChromaDB Or InterSystems IRIS](#chromadb-or-intersystems-iris)
6. [Patient Chat](#patient-chat)
7. [Synthetic Data And Evaluation](#synthetic-data-and-evaluation)
8. [Public Release](#public-release)
9. [Citation](#citation)
10. [License](#license)
11. [Clinical Use](#clinical-use)

## Quick Start

Python 3.10 or newer, a local Ollama server, and a spaCy biomedical model are required
for extraction. Model downloads require internet access; the validation and manual
configuration stages do not download models or send notes anywhere.

```bash
git clone --branch main --single-branch https://github.com/pgsalome/oncorag.git
cd oncorag
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
ollama pull phi3:mini

oncorag --config configs/oncorag_synthetic_english.json --stage validate
oncorag --config configs/oncorag_synthetic_english.json
```

The spaCy model must match your installed spaCy version; check `python -m spacy validate`.
The current biomedical models and default embeddings are English-oriented. German
and mixed notes are retained verbatim with their language metadata; this does not
establish equivalent extraction accuracy across languages.

From a checkout, the equivalent command is:

```bash
python scripts/run_oncorag_full_pipeline.py --config configs/oncorag_synthetic_english.json
```

For an Ollama server configured on a different port, use explicit overrides. For
example, this runs the mixed-language fixture against a server on port 11435:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 OLLAMA_MODEL=phi3:mini \
  .venv/bin/python scripts/run_oncorag_full_pipeline.py \
  --config configs/oncorag_synthetic_mixed.json
```

Use `oncorag_synthetic_german.json` for German. Mixed fixtures contain both English
and German reports **within every patient**. CLI overrides take precedence over
`OLLAMA_HOST` / `OLLAMA_MODEL`, which take precedence over JSON configuration.

## Your Variables

Start with `examples/features.synthetic.yaml`:

```yaml
features:
  - name: latest_hemoglobin
    type: numeric
    expected_range: [0, 30]
    unit: g/dL
    synonyms: [hemoglobin, Haemoglobin]
    description: Hemoglobin in g/dL from the most recent dated report.
  - name: treatment
    type: categorical
    expected_range: [chemotherapy, radiotherapy]
    description: The treatment actually started, not merely planned.
```

Types: `integer`, `numeric`, `boolean`, `date`, `categorical`, `ordinal`, `string`.
Numeric bounds are inclusive; categorical values are validated against complete
labels. Missing values are JSON `null`, distinct from invalid output and service
errors. Dates are calendar-validated ISO `YYYY-MM-DD`. Describe temporal selection,
units and source preferences explicitly; supply multilingual synonyms when useful.

`features.configuration_mode: manual` creates deterministic configs from these
definitions. It does **not** claim ontology enrichment. `automatic` uses the existing
LLM/UMLS enrichment workflow; it additionally needs `UMLS_API_KEY`, optional
`BIOPORTAL_API_KEY`, WordNet resources (`python -m nltk.downloader wordnet omw-1.4`),
and network access for concept lookup. Only feature definitions, not patient notes,
belong in ontology requests. Generated configs can be inspected before extraction:

```bash
oncorag --config configs/oncorag_synthetic_english.json --stage config
python oncoraggraph/create_config.py --mode manual \
  --features-file examples/features.synthetic.yaml \
  --output-dir generated/custom --language english
```

## Your Notes

Set exactly one of `inputs.notes_root` or `inputs.registry_path`.

```text
notes/
  patient-001/
    oncology/2024-01-12.txt
    radiology/2024-02-03__report-02.txt
```

Same-date reports use `YYYY-MM-DD__unique-note-id.txt`. Alternatively use CSV,
JSONL or JSON (`[{...}]` or `{"notes": [...]}`) registry records:

```json
{"patient_id":"patient-001","note_id":"report-02","report_type":"radiology","date":"2024-02-03","language":"de","path":"notes/report-02.txt"}
```

Registry paths resolve relative to the registry. Config paths resolve relative to
the JSON config, regardless of working directory. Dates and report types come from
the path or registry, never guesses from report text. Invalid dates, empty notes,
duplicate IDs and malformed layouts fail validation. Optional `patient_ids_file`
contains one exact patient ID per line, including leading zeros.

## Parameters And Outputs

Copy a synthetic config and change its input, feature and output paths. The expanded
`configs/oncorag_full_pipeline.example.json` also records the paper's evaluation
protocol. Deployment settings are adjustable, not forced to the paper's values.

- `runtime.ollama`: host, model, temperature, context window, timeout, output limit
  and `validation_retries` (default 1). Invalid types/quotes get a bounded repair
  attempt using the same evidence, never reference answers; every attempt is saved.
  Structured generation restricts citations to retrieved text and note IDs, then
  independently checks the quote/source pairing and declared value constraints.
- `runtime.random_seed`: generation seed. The portable runner is currently serial;
  `workers` must be 1. Model/hardware determinism is not guaranteed.
- `retrieval`: candidate entity limit, graph depth, exact final top-k, six scoring
  weights, diffusion threshold/neighbors/iterations/residual mixing.
- `graph`: model configurations, context filters, deduplication and sentence inclusion.
- `temporal_anchoring`: additional instructions passed to the extraction prompt.
  This is an LLM policy, not a deterministic date-filtering guarantee.

The portable graph augments the entity graph with source sentence nodes by default,
preserving numeric and multilingual facts missed by NER. Set
`graph.include_report_sentences: false` for entity-only construction. This portable
mode is not an assertion of exact reproduction of every paper experiment.

`--stage graph` stops after graph construction. Extraction writes one JSON graph per
patient/content fingerprint, per-patient results, aggregate `structured_features.json`,
the effective parameters, and prompt/evidence records under `outputs.root`. Graph
caches are invalidated by changed notes or graph settings; vectors are replaced per
patient and result/prompt files are regenerated. `--force-rebuild` rebuilds graphs.
Outputs and logs may contain sensitive clinical information: keep them private.

## ChromaDB Or InterSystems IRIS

ChromaDB is the default local persistent store. To use IRIS:

```bash
pip install -e '.[iris]'
export IRIS_USERNAME=your_database_user
export IRIS_PASSWORD=your_database_password
```

Copy the `vector_store` settings from `configs/vector_store.iris.example.yaml` into
your pipeline JSON, then set `backend: iris`. Configure host, port, namespace, table
and embedding dimension. The default SapBERT vectors have 768 dimensions. Database
credentials are read from the environment; do not place passwords in configs.
`--vector-backend iris` overrides the selected backend. Each cohort/patient has an
isolated collection. IRIS failures do not silently fall back to ChromaDB. Live IRIS
requires a reachable server and permissions to initialize the configured table.

## Patient Chat

Use the same pipeline JSON and feature definitions for questions about one patient.
Chat generates any missing feature configs, builds or reuses that patient's graph,
and indexes it using the configured ChromaDB or IRIS backend. No extraction run is
required first. The terminal interface is included in the base installation:

```bash
oncorag-chat --config configs/oncorag_synthetic_mixed.json --list-patients
oncorag-chat --config configs/oncorag_synthetic_mixed.json \
  --patient-id SYN-DEMO-001 --loop
oncorag-chat --config configs/oncorag_synthetic_mixed.json \
  --patient-id SYN-DEMO-001 --question "What treatment actually started?" --json
```

From a checkout, `python run_chatbot.py` accepts the same arguments. Follow-up
questions use bounded conversation history; `/clear` forgets it and `/quit` exits.
The same Ollama environment variables and CLI overrides apply as for extraction.
Optional settings in the pipeline JSON are:

```json
{
  "chat": {
    "history_turns": 5,
    "max_question_chars": 4000,
    "max_history_chars": 12000,
    "feature_match_threshold": 0.45
  }
}
```

For the browser interface, run from the checkout:

```bash
pip install -e '.[chat]'
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 -- \
  --config configs/oncorag_synthetic_mixed.json
```

Open `http://127.0.0.1:8501` and select a patient. The interface shows source quotes,
report dates/types/languages, and supported measurement timelines. Ontology
definitions appear separately from patient evidence. Patient or configuration
changes clear the conversation, including when loading the next patient fails.
History is held in session memory; chat does not automatically save transcripts.
Graph/vector caches still contain source clinical information. This local interface
has no application authentication: do not expose it to a network or share screenshots,
logs, JSON answers or caches containing clinical data without appropriate controls.

Answers require quotations checked against retrieved original notes. That verifies
source provenance, **not** whether every generated claim follows from its citation.
Timelines are parsed separately from retrieved notes and remain available when the
model abstains from a narrative answer; an abstention is still reported as `missing`.
Missing evidence, invalid responses and backend errors are distinct statuses.
Review answers and timelines against the notes; this is not clinical decision support.

## Synthetic Data And Evaluation

The public release bundles only small purpose-authored English/German/mixed
regression fixtures, with 3 patients, 9 notes and 12 typed reference answers each.
Every mixed-fixture patient has both English and German reports. These are smoke
tests, not clinical validation or held-out accuracy estimates.

The larger template-derived English and German cohorts are **not distributed**
with this release: their provenance and text/template redistribution review remain
pending. The local export tools are retained for separately authorized data.
See [dataset provenance](examples/datasets/README.md).

```bash
pip install -e '.[dev,chat]'
python -m pytest tests -q
python scripts/run_synthetic_smoke.py --ollama-host http://127.0.0.1:11434
python scripts/run_chat_smoke.py --ollama-host http://127.0.0.1:11434
python scripts/evaluate_synthetic.py \
  --config configs/oncorag_synthetic_mixed.json \
  --results outputs/synthetic_mixed/structured_features.json \
  --output outputs/synthetic_mixed/evaluation.json
python scripts/evaluate_synthetic.py \
  --config configs/oncorag_synthetic_mixed.json \
  --write-experiments outputs/experiments
```

Evaluation checks all expected patient/feature pairs, typed exact match, categorical
macro-F1, patient-bootstrap intervals, confidence groups and configured strata.
Missing/error predictions are not silently excluded. Experiment configs cover
top-k, weight perturbations and model/context combinations; generating them does
not run the experiments. Remaining paper analyses (including clinical baselines,
inter-rater comparisons and full-cohort validation) need appropriate study data.

Unit/integration tests use controlled model doubles and real temporary Chroma stores.
An optional live IRIS test runs with `ONCORAGGRAPH_TEST_IRIS=1` and credentials.
The chat smoke test checks nine real-model turns across English, German and
same-patient mixed notes, including follow-up dates, patient switching and source
quotes. These narrow fixtures do not establish general conversational accuracy.

## Public Release

The canonical repository is [pgsalome/oncorag](https://github.com/pgsalome/oncorag),
which retains the existing [InterSystems OpenExchange listing](https://openexchange.intersystems.com/package/oncorag).
The release branch preserves the original March 2025 project commit while excluding
intermediate research history.

Do not publish a research checkout or its Git history wholesale. To create an
allowlisted source snapshot without Git history for review:

```bash
python scripts/prepare_public_release.py --destination public_release/review-new
```

`--include-datasets` also stages the full exports for local review, without asserting
redistribution rights. The export omits research outputs, clinical configuration,
caches and old Git history. It substitutes portable system defaults and writes a
hash manifest. Nothing is committed or pushed by this command.
Before making an existing repository public, audit its branches, tags and historical
content, and remove any sensitive data from its history. Adding a clean branch or
changing the default branch does not remove other branches or older commits.

## Citation

If you use OncoRAG, please cite the associated paper:

Salome P, Knoll M, Walz D, et al. [OncoRAG: graph-based retrieval enabling
clinical phenotyping from oncology notes using local mid-size language
models](https://doi.org/10.1038/s41746-026-03170-8). *npj Digital Medicine*.
2026. doi: `10.1038/s41746-026-03170-8`.

## License

This software is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

Noncommercial use, modification, and redistribution are permitted subject to
the license terms. Commercial use requires a separate written agreement.

This is not an OSI-approved open-source license. Dataset and model rights are
separate and may impose additional conditions.

## Clinical Use

This repository contains research software. It is not a clinically validated
decision-support system. Review extracted values against their source evidence
before downstream use. The paper's progression-free-survival analysis is downstream
evaluation, not a built-in clinical prediction tool.
