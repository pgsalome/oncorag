# Synthetic Datasets

The oncorag-e (English) and oncorag-d (German) cohorts are included in this
repository, along with all notes, relative registries, projected event labels,
patient splits and file manifests. No separate data download is needed.

| Directory | Patients | Notes | Annotation scope |
| --- | ---: | ---: | --- |
| `oncorag-e` | 489 | 2,930 | 5,761 note-level CTCAE events |
| `oncorag-d` | 489 | 2,930 | 5,987 note-level toxicity events |
| `demo/english` | 3 | 9 | 12 typed patient-feature answers |
| `demo/german` | 3 | 9 | 12 typed patient-feature answers |
| `demo/mixed` | 3 | 9 | 12 typed patient-feature answers |

Full English and German cohorts contain different generated patients. Do not merge
unrelated patients to construct bilingual timelines. The three small synthetic
example cohorts are parallel versions of the same three synthetic patients; every
patient in the mixed-language cohort has both German and English notes in one timeline.

## Run The Full Cohorts

After installing OncoRAG and its extraction models, run from the repository root:

```bash
python scripts/run_oncorag.py --config configs/oncorag-e.json
python scripts/run_oncorag.py --config configs/oncorag-d.json
```

Use `--stage validate` to check all inputs without contacting model services.
The full-cohort feature lists are [oncorag-e](../features.oncorag-e.yaml) and
[oncorag-d](../features.oncorag-d.yaml). Replace them with your own variables as
needed. The examples use deterministic manual feature configuration and separate
graph/vector/output namespaces. They intentionally have no `evaluation.gold_path`:
event labels are not answers to arbitrary patient-level questions.

English examples target explicitly documented visit dates, treatment weeks and
functional limitations. German examples target diagnosis dates, radiotherapy doses
and explicitly documented laterality. The full notes do not consistently document
the age and hemoglobin variables used in the small synthetic example cohorts.

## Files And Metadata

A single canonical copy of each note is stored under
`notes/<patient_id>/<report_type>/<date>__<note_id>.txt` within each dataset.
The suffix preserves multiple reports of the same type on the same date.

`registry.csv` supplies patient ID, note ID, report type, ISO date, language, and
a path relative to the registry directory. Folder paths and registry metadata
describe the same documents. Use the registry when note-level language and stable
annotation note IDs are required. `manifest.json` records counts, SHA-256 file
hashes, provenance and scope.

Patient IDs use `oncorag-e-0001` or `oncorag-d-0001`; report IDs use
`oncorag-e-note-00001` or `oncorag-d-note-00001`. These names are consistent across
folders, registries, labels and report headers. Existing split membership is
preserved when identifiers are renamed.

Full-cohort `labels.jsonl` contains only note metadata and selected term, grade,
negation and temporal event labels. Missing negation remains absent. Private
source-style identifiers, demographics, history payloads, local source paths and
upstream evidence snippets are not included. Each note is explicitly marked
synthetic; the German derivative removes a real institution header.

`splits.json` assigns each patient to exactly one split using seed 42: 342 training,
73 development and 74 test patients per full cohort. These are patient-disjoint
partitions of template-generated data, not independent clinical validation cohorts.
Templates are shared across splits. Splits do not filter pipeline inputs
automatically; use `--patient-ids-file` for a selected subset.

## Small Synthetic Example Cohorts

The small synthetic example cohorts are purpose-authored versions of the same
three synthetic timelines, with no upstream patient-derived text. They share dates,
facts, note identities and gold answers. Their dataset IDs are `demo_english`,
`demo_german` and `demo_mixed`. Isolate them by dataset ID in caches; do not count
the language variants as independent subjects.

Use [features.synthetic.yaml](../features.synthetic.yaml) for these four variables:

- `diagnosis_date`: explicit initial diagnosis date.
- `age_at_diagnosis`: explicit age in whole years.
- `treatment_name`: started treatment, normalized to an English enum label.
- `latest_hemoglobin`: value in g/dL from the latest report date.

`gold.jsonl` has one row per patient and feature, with a typed `value` and
`evidence_note_ids`. These cases check ingestion, temporal selection, multilingual
normalization and output types. They are software regression tests, not a clinical
accuracy benchmark.

## Provenance And Limitations

The [technical provenance review](PROVENANCE.md) explains the source generators,
privacy projection, exact reviewed versions and known label-quality limitations.
English notes are CTCAE template-generated text with upstream Synthea metadata,
not verbatim Synthea note exports. German notes come from a standalone seeded
template generator, not the Synthea engine. Neither cohort establishes clinical
grade correctness, real-world language coverage or clinical population realism.

## Recreating An Export

Users can work directly with the bundled files. Maintainers with the separately
authorized original inputs can recreate the exact reviewed derivatives:

```bash
python scripts/prepare_reviewed_cohorts.py \
  --english-source /path/to/english-source \
  --german-source /path/to/german-source \
  --output-root /path/to/new/versioned-output
```

This wrapper projects fields, checks fixed fingerprints of both audited inputs,
adds the synthetic notices, applies the public identifiers and writes new
`oncorag-e` and `oncorag-d` directories with their manifests. Changed inputs require
a new review. No clinical template assets or private source files are
bundled to make this command self-contained.

The generic `scripts/export_synthetic_datasets.py` remains available for separately
authorized local inputs. Its marker checks alone do not approve data for release.
Regenerating only the small synthetic example cohorts needs no upstream data:

```bash
python scripts/export_synthetic_datasets.py \
  --demo-only --output-root /path/to/new/demo-output
```

## Use And Citation

The OncoRAG synthetic-note derivatives and release metadata are distributed under
the repository's [PolyForm Noncommercial License](../../LICENSE). The original
terminology and generator attribution is described in [PROVENANCE.md](PROVENANCE.md);
the repository does not relicense third-party terminology or source datasets.

When using these cohorts, cite the [OncoRAG paper](https://doi.org/10.1038/s41746-026-03170-8)
and record the dataset ID, repository commit and manifest hash. These community
datasets are not the paper's clinical evaluation cohorts. No separate dataset DOI
has been assigned.
