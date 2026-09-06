# Synthetic Cohort Provenance

Review ID: `oncorag-cohort-provenance-v1`, 2026-09-05.

This technical review covers source lineage, released files and reproducibility.
The public notes carry synthetic notices. Clinical inputs, source-style patient
identifiers and private provenance metadata are excluded.

## oncorag-e (English)

Source: `hybrid_synthea_ctcae_phase2`, with 489 patients, 2,930 notes and 5,761
annotated events. Its builder copies CTCAE-generated note bodies byte-for-byte
and adds Synthea encounter and demographic metadata. The public export omits
that metadata and source-style identifiers carried by the original labels.

The upstream configuration includes style information mined from clinical notes.
The current renderer uses authored templates and narrative rules. Mined phrase
choices affect its random sequence without inserting those phrases into notes.
Saved metadata records Ollama as disabled.

Targeted checks found no known source-style identifiers or matches to 49 selected
source phrases in the 2,930 notes. An exact historical replay was not performed
because the generator has since changed. Code inspection and limited string
checks leave broader provenance and privacy questions open.

## oncorag-d (German)

Source: `ricci_rhgg_termgrade_longitudinal`, with 489 patients, 2,930 notes and
5,987 annotated events. All 5,865 source files were reproduced byte-for-byte
using the standalone standard-library generator, seed 42 and recorded settings.
The replay used fixed templates, with no clinical files, model calls or Synthea.

Preparation documentation cites clinical-cohort schema inspiration. The
generator lacks usable source-control history; template authorship remains
unverified. A comparison of 120 long template literals with the referenced schema
CSV found no complete normalized matches.

The public export removes the hospital heading, adds a synthetic notice, and
excludes duplicated note text, local paths and private history/provenance objects.

## Known Annotation Limits

- Event terms and grades are template-assigned and lack clinician adjudication.
  Term/grade compatibility and clinical realism need independent assessment.
- In German, 1,498 events have supplementary evidence absent from the note.
  Main event spans are present. The public labels omit all source evidence snippets.
- Events are sampled without a complete clinical trajectory model. German
  pre-treatment notes may include future therapy dates; a mentioned date can
  describe a plan.
- Full-cohort labels describe note-level events. Patient-level reference answers
  are available only for the small example cohorts.
- Templates are shared across patient splits, limiting generalization estimates.
  English and German contain different generated populations.

## Reviewed Files

Public cohort, patient and report identifiers were renamed consistently across
files and metadata. Clinical content, dates, labels and split membership were
preserved. The input hashes below cover projected notes, relative registries,
selected label fields and splits, excluding private source identifiers and metadata.

| Projected input | SHA-256 |
| --- | --- |
| English | `be2737652e94d01e37f6d0d4c318cb35900216fe3579b5674faedc5d1c7e044a` |
| German | `def4936409b0b7aaf74c4046b920797f915a22f67350d149f22c12ab2b2106da` |

`scripts/prepare_reviewed_cohorts.py` requires these input hashes.
`scripts/prepare_public_release.py --include-datasets` checks output manifests
and every file hash. Dataset manifests record transformations and review scope.

Reviewed source-code SHA-256 identifiers:

- Hybrid English builder: `9ad6dd22bb546ed556ac071d60ae3900b4a19c6903a19cf1c2f38b6532c15259`.
- Current CTCAE note generator: `7c5e4cb8962b946762843981f93d9146dfef4a41a00cf9f68f0e22f45a92cb8d`.
- Standalone German generator: `d21a0d5dba063b94e2be2e22a0641b0d5d5c9fd030c1e557f8415fb8f0faceaf`.

## Attribution And Rights

The [repository license](../../LICENSE) covers OncoRAG-authored synthetic-note
derivatives and release metadata. Historical template ownership and any required
institutional approvals need separate review. Rights to original clinical records
and restricted source datasets remain with their respective owners.

The English generator used [Synthea](https://github.com/synthetichealth/synthea)
for metadata. Synthea's [Apache License 2.0](https://github.com/synthetichealth/synthea/blob/master/LICENSE)
applies to its software; separately authored note templates have their own rights.

Event labels reference the National Cancer Institute's
[CTCAE resources](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events).
[CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf)
incorporates MedDRA terminology. Complete terminology tables, instruments and
clinical source templates are excluded. NCI's
[reuse policy](https://www.cancer.gov/policies/copyright-reuse) distinguishes its own
material from separately protected third-party content. Those rights apply
separately from the repository license.
