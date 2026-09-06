# Synthetic Datasets

The repository includes two full synthetic cohorts with notes, registries,
event labels and patient splits:

| Dataset | Patients | Notes |
| --- | ---: | ---: |
| `oncorag-e` (English) | 489 | 2,930 |
| `oncorag-d` (German) | 489 | 2,930 |

The cohorts contain different generated patients. Small English, German and mixed
datasets are also included for testing.

## Run The Full Cohorts

After [installing OncoRAG](../../README.md#quick-start), run:

```bash
python scripts/run_oncorag.py --config configs/oncorag-e.json
python scripts/run_oncorag.py --config configs/oncorag-d.json
```

Add `--stage validate` to check inputs locally. Edit the feature lists to choose
your variables:

- [oncorag-e](../features.oncorag-e.yaml): visit dates, treatment weeks and functional limitations.
- [oncorag-d](../features.oncorag-d.yaml): diagnosis dates, radiotherapy doses and laterality.

Both configurations use manual feature configuration and separate output folders.
Full-cohort labels describe events within notes. Patient-level evaluation requires
reference answers for the variables you choose.

## Files And Metadata

Notes are stored as `notes/<patient_id>/<report_type>/<date>__<note_id>.txt`.
The note ID distinguishes reports of the same type and date.

- `registry.csv`: patient ID, note ID, report type, ISO date, language and note path
  relative to the registry. Use it to preserve note IDs and language metadata.
- `labels.jsonl`: note-level events with terms, grades, negation and temporality
  where available. Missing negation is left unspecified.
- `manifest.json`: dataset counts, SHA-256 file hashes and provenance.
- `splits.json`: 342 training, 73 development and 74 test patients per full cohort,
  assigned with seed 42. Each patient belongs to one split; templates are shared
  across splits.

To select a split, pass a file with one patient ID per line using
`--patient-ids-file`. Patient IDs follow `oncorag-e-0001` or `oncorag-d-0001`;
note IDs follow `oncorag-e-note-00001` or `oncorag-d-note-00001`.

## Small Test Datasets

`demo/english`, `demo/german` and `demo/mixed` each contain 3 synthetic patients,
9 notes and 12 typed reference answers. They are language variants of the same
patients, dates and clinical facts. In `mixed`, every patient has English and
German notes in one timeline.

Use dataset IDs `demo_english`, `demo_german` and `demo_mixed` to keep their caches
separate. These variants represent three unique patients in total.

Use [features.synthetic.yaml](../features.synthetic.yaml) for these four variables:

- `diagnosis_date`: explicit initial diagnosis date.
- `age_at_diagnosis`: explicit age in whole years.
- `treatment_name`: started treatment, normalized to an English enum label.
- `latest_hemoglobin`: value in g/dL from the latest report date.

`gold.jsonl` contains one reference answer per patient and feature, with a typed
`value` and `evidence_note_ids`. The tests check ingestion, temporal selection,
multilingual normalization and output types.

## Use And Citation

The OncoRAG synthetic notes and release metadata use the
[PolyForm Noncommercial License](../../LICENSE). Third-party terminology and source
licenses apply separately; see [attribution and provenance](PROVENANCE.md).

Cite the [OncoRAG paper](https://doi.org/10.1038/s41746-026-03170-8) and record the
dataset ID, repository commit and manifest hash. The paper reports results on
separate clinical cohorts.
