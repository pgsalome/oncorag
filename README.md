# OncoRAG

[![Paper](https://img.shields.io/badge/npj%20Digital%20Medicine-10.1038%2Fs41746--026--03170--8-blue)](https://www.nature.com/articles/s41746-026-03170-8)
[![License](https://img.shields.io/badge/use-noncommercial_only-4B5563)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)

Clinical phenotyping through structured feature extraction from oncology notes.

Define your clinical variables, build a knowledge graph for each patient, and
extract structured features with source evidence. OncoRAG combines ontology
enrichment, graph-diffusion reranking and structured prompting with local language
models.

With the supplied local configurations, notes stay local.

An optional patient chatbot uses the same graphs and retrieval configuration.

Paper: [OncoRAG in npj Digital Medicine](https://www.nature.com/articles/s41746-026-03170-8),
DOI `10.1038/s41746-026-03170-8`.

[![OncoRAG study workflow: clinical notes, configuration, extraction, retrieval, generation, and downstream prediction](graphicalabstract.png)](graphicalabstract.png)

*OncoRAG workflow and the paper's downstream progression-free-survival analysis.*

## Table Of Contents

1. [Synthetic Data And Evaluation](#synthetic-data-and-evaluation)
2. [Quick Start](#quick-start)
3. [Your Variables](#your-variables)
4. [Your Notes](#your-notes)
5. [Parameters And Outputs](#parameters-and-outputs)
6. [ChromaDB Or InterSystems IRIS](#chromadb-or-intersystems-iris)
7. [Patient Chat](#patient-chat)
8. [Public Release](#public-release)
9. [Citation](#citation)
10. [License](#license)
11. [Clinical Use](#clinical-use)

## Synthetic Data And Evaluation

The repository includes two full synthetic cohorts and three small example cohorts.

| Dataset | Patients | Notes | Labels |
| --- | ---: | ---: | --- |
| [English](examples/datasets/english/registry.csv) | 489 | 2,930 | 5,761 note-level CTCAE events |
| [German](examples/datasets/german/registry.csv) | 489 | 2,930 | 5,987 note-level toxicity events |
| English, German and mixed examples | 3 per variant | 9 per variant | 12 typed reference answers per variant |

The full English and German cohorts contain different generated patients. In the
small mixed cohort, each patient has both English and German notes.

These datasets support software testing and experimentation. The paper reports
results on separate clinical cohorts. Annotation quality and language coverage are
described in the [dataset documentation](examples/datasets/README.md) and
[provenance report](examples/datasets/PROVENANCE.md).

After [Quick Start](#quick-start), run either full cohort with its own feature list:

```bash
python scripts/run_oncorag.py --config configs/oncorag_synthetic_english_full.json
python scripts/run_oncorag.py --config configs/oncorag_synthetic_german_full.json
```

Add `--stage validate` to check the input files and feature definitions. Each
configuration includes a feature list matched to its cohort. Full-cohort labels
describe note-level events; the small cohorts provide patient-level reference
answers for four variables.

The following evaluation commands use the small synthetic example cohorts:

```bash
pip install -e '.[dev,chat]'
python -m pytest tests -q
python scripts/run_synthetic_smoke.py --ollama-host http://127.0.0.1:11434
python scripts/run_chat_smoke.py --ollama-host http://127.0.0.1:11434
python scripts/evaluate_synthetic.py \
  --config configs/oncorag_synthetic_mixed.json \
  --results outputs/synthetic_smoke/mixed/structured_features.json \
  --output outputs/synthetic_smoke/mixed/evaluation.json
python scripts/evaluate_synthetic.py \
  --config configs/oncorag_synthetic_mixed.json \
  --write-experiments outputs/experiments
```

Evaluation covers every expected patient-feature pair, including missing and failed
predictions. It reports typed exact match, categorical macro-F1, patient-bootstrap
confidence intervals and results by confidence group and configured stratum.

`--write-experiments` creates configurations for top-k, retrieval weights, models
and context windows. Run the generated configurations separately. Reproducing the
paper's clinical comparisons requires the corresponding study data.

The test suite uses simulated model responses and temporary ChromaDB databases.
Set `ONCORAGGRAPH_TEST_IRIS=1` and supply credentials to include a live IRIS test.
The chat test script runs nine local-model turns covering both languages,
follow-up dates, patient switching and source quotations.

## Quick Start

Requirements: Python 3.10 or newer, a local Ollama server, and a spaCy biomedical
model.

Install OncoRAG and download the required models:

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

Check spaCy model compatibility with `python -m spacy validate`. The default
biomedical and embedding models are English-oriented; assess performance on your
target language and cohort.

To run a small example cohort with Python:

```bash
# English
python scripts/run_oncorag.py --config configs/oncorag_synthetic_english.json

# German
python scripts/run_oncorag.py --config configs/oncorag_synthetic_german.json

# English and German notes within each patient
python scripts/run_oncorag.py --config configs/oncorag_synthetic_mixed.json
```

## Your Variables

Start with `examples/features.synthetic.yaml`:

```yaml
features:
  - name: latest_hemoglobin
    type: numeric
    expected_range: [0, 30]
    unit: g/dL
    description: Hemoglobin in g/dL from the most recent dated report.
  - name: treatment
    type: categorical
    expected_range: [chemotherapy, radiotherapy]
    description: Cancer treatment documented as started.
```

Types: `integer`, `numeric`, `boolean`, `date`, `categorical`, `ordinal`, `string`.
Numeric bounds are inclusive. Categorical outputs must match an allowed label.
Use ISO `YYYY-MM-DD` dates and JSON `null` for missing values. Invalid outputs and
service errors have separate statuses. Include units, temporal selection and
source preferences in the description. An optional `synonyms` list accepts extra
terms in any language.

Set `features.configuration_mode` to choose how feature configurations are built:

- `automatic`: `create_config.py` generates synonyms and enriches definitions
  through a language model and ontology lookup.
- `manual`: builds configurations directly from your definitions and supplied
  terms. The synthetic examples use this mode.

Automatic enrichment requires `UMLS_API_KEY`, WordNet resources
(`python -m nltk.downloader wordnet omw-1.4`) and internet access for concept
lookup. `BIOPORTAL_API_KEY` is optional. Keep patient information out of feature
definitions, which are used in ontology queries.

Generate and inspect feature configurations before extraction:

```bash
oncorag --config configs/oncorag_synthetic_english.json --stage config
python oncoraggraph/create_config.py --mode manual \
  --features-file examples/features.synthetic.yaml \
  --output-dir generated/custom --language english
```

Generated rules, examples and category mappings guide extraction. Your feature
definition controls the output type and allowed values; supporting quotations
come from the patient's notes.

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

Note paths are relative to the registry file; configuration paths are relative to
the JSON configuration file. Folder names or registry fields supply report dates
and types. Validation checks dates, note contents, unique IDs and folder structure.
To select patients, use `patient_ids_file` with one exact ID per line, preserving
leading zeros.

## Parameters And Outputs

Start with a synthetic configuration and update the input, feature and output
paths. `configs/oncorag_full_pipeline.example.json` contains the expanded
parameters and the paper's evaluation protocol. Adjust settings for your data,
models and hardware.

Set the Ollama host and model in the JSON config, or choose them for one run with
`--ollama-host` and `--ollama-model`. For example, to use a local server on port 11435:

```bash
python scripts/run_oncorag.py --config configs/oncorag_synthetic_mixed.json \
  --ollama-host http://127.0.0.1:11435 --ollama-model phi3:mini
```

You can also set `OLLAMA_HOST` and `OLLAMA_MODEL` in your environment. Command-line
values are used first, then environment variables, then the JSON settings.

- `runtime.ollama`: host, model, temperature, context window, timeout, output limit
  and `validation_retries` (default 1). Failed type or quotation checks can be
  retried up to this limit using the same source evidence. Every attempt is saved.
- `runtime.random_seed`: generation seed. Patients are processed sequentially;
  set `workers` to 1. Results can vary across models and hardware.
- `retrieval`: candidate entity limit, graph depth, exact final top-k, six scoring
  weights, and graph-diffusion reranking settings (threshold, neighbors, iterations
  and residual mixing).
- `graph`: model configurations, context filters, deduplication and sentence inclusion.
- `temporal_anchoring`: temporal instructions for the language model to apply
  during extraction.

Graphs include source sentences alongside recognized entities, retaining numeric
and multilingual information. Set `graph.include_report_sentences: false` to
build entity-only graphs.

Use `--stage graph` to stop after graph construction. Extraction results under
`outputs.root` include patient graphs, per-patient results, combined
`structured_features.json`, run parameters, prompts and source evidence. Changed
notes or graph settings invalidate cached graphs. Each extraction run replaces
patient vectors and regenerates result and prompt files. Use `--force-rebuild`
to rebuild graphs explicitly.

## ChromaDB Or InterSystems IRIS

ChromaDB is the default local persistent store. To use IRIS:

```bash
pip install -e '.[iris]'
export IRIS_USERNAME=your_database_user
export IRIS_PASSWORD=your_database_password
```

Copy the `vector_store` settings from `configs/vector_store.iris.example.yaml` into
your pipeline JSON and set `backend: iris`. Configure the host, port, namespace,
table and embedding dimension. The default SapBERT vectors have 768 dimensions.
Keep credentials in environment variables.

`--vector-backend iris` selects IRIS for a run. Collections are separated by cohort
and patient. IRIS connection failures stop the run; the server must be reachable
and the account must have permission to initialize the configured table.

`runtime.local_processing_only` defaults to `true` and requires loopback addresses
for both Ollama and IRIS in extraction and chat. Using an approved remote service
requires setting it to `false` and configuring that service's data protection.

## Patient Chat

Chat uses the same configuration, feature definitions and patient graphs as
structured extraction. It can also build a patient's graph directly. The terminal
interface is included in the base installation:

```bash
oncorag-chat --config configs/oncorag_synthetic_mixed.json --list-patients
oncorag-chat --config configs/oncorag_synthetic_mixed.json \
  --patient-id SYN-DEMO-001 --loop
oncorag-chat --config configs/oncorag_synthetic_mixed.json \
  --patient-id SYN-DEMO-001 --question "What treatment actually started?" --json
```

From the repository folder, `python run_chatbot.py` accepts the same arguments.
Follow-up questions use recent conversation history; `/clear` clears it and
`/quit` exits. Ollama settings follow the same precedence as extraction.
Optional conversation settings are:

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

For the browser interface, run from the repository folder:

```bash
pip install -e '.[chat]'
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 -- \
  --config configs/oncorag_synthetic_mixed.json
```

Open `http://127.0.0.1:8501` and select a patient. Answers show source quotations and
report dates, types and languages. Measurement timelines are built from retrieved
notes and remain available when the model cannot answer. Ontology definitions
have their own section.

Quotations are checked against the source notes. Review the clinical interpretation
of each answer against that evidence. Responses report missing evidence, invalid
output and backend errors separately.

Conversation history stays in session memory and clears when you change patient
or configuration. Keep the browser app on localhost. Network deployment requires
separate authentication and access controls.

## Public Release

The canonical repository is [pgsalome/oncorag](https://github.com/pgsalome/oncorag),
which retains the existing [InterSystems OpenExchange listing](https://openexchange.intersystems.com/package/oncorag).
The release branch preserves the original March 2025 project commit while excluding
intermediate research history.

Create a release directory containing the approved source files and datasets:

```bash
python scripts/prepare_public_release.py --destination public_release/review-new --include-datasets
```

`--include-datasets` adds the full cohorts after checking their reviewed file
hashes. The default export includes the small example cohorts. The exporter copies
approved files, applies portable defaults and writes a file manifest. Git
publication is a separate step.

Before publication, review all branches, tags and repository history for sensitive
data. The export excludes research outputs, clinical configurations, caches and
Git history.

## Citation

If you use OncoRAG, please cite the associated paper:

Salome P, Knoll M, Walz D, et al. [OncoRAG: graph-based retrieval enabling
clinical phenotyping from oncology notes using local mid-size language
models](https://doi.org/10.1038/s41746-026-03170-8). *npj Digital Medicine*.
2026. doi: `10.1038/s41746-026-03170-8`.

## License

OncoRAG is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE),
which permits noncommercial use, modification and redistribution under its terms.
Commercial use requires a separate written agreement. Dataset and model licenses
apply separately.

## Clinical Use

OncoRAG is research software. Clinical use requires independent validation and
appropriate oversight. Review extracted features and their source evidence before
using the results. When working with patient records, protect the graphs, vector
stores, prompts, logs and exported answers as clinical data.
