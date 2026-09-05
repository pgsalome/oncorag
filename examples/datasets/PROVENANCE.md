# Synthetic Cohort Provenance Review

Review ID: `oncorag-cohort-provenance-v1`, 2026-09-05.

This is a technical source, payload and reproducibility review. It is not legal
certification, a guarantee against every possible identifier, or clinical
validation. The release retains reviewed synthetic note bodies with explicit
synthetic notices, projects label fields and excludes original clinical inputs,
source-style identifiers and private provenance payloads.

## English

The `hybrid_synthea_ctcae_phase2` source contains 489 synthetic patients, 2,930
notes and 5,761 annotated events. The hybrid builder copies all 2,930 upstream
CTCAE-generated note bodies byte-for-byte and adds Synthea encounter/demographic
metadata. That additional metadata is not included in the public projection.

The upstream configuration contains style information mined from clinical notes.
Every original label carries a source-style patient identifier; these fields must
not be distributed. The current renderer's text-producing path uses template
surface forms and authored narrative rules, not the mined section-rendering
functions. Mined auxiliary phrase choices affect random-number consumption but
their strings are not interpolated into the note bodies.

Targeted checks found no known source-style identifiers in any of the 2,930 notes.
None of 49 source-matching phrases of four or more words, absent from the authored
generator/surface literals, occurred in the notes. Saved metadata has Ollama
disabled. The separate source-snippet generator is not the generator identified
by this cohort's metadata. Its raw outputs are not part of this release.

An exact historical full-generation replay was not performed: the current
generator contains later changes. Phrase checks are targeted, not exhaustive
proof of historical authorship. Fixed payload fingerprints prevent a different
source revision from silently inheriting this review.

## German

The `ricci_rhgg_termgrade_longitudinal` source contains 489 synthetic patients,
2,930 notes and 5,987 annotated events. All 5,865 original artifacts, including
notes, labels and summary files, were reproduced byte-for-byte from the standalone
standard-library generator, seed 42 and its recorded scalar settings. The replay
read no clinical files and invoked neither a model nor Synthea.

The code samples patient histories and events from fixed templates. All 489
patient histories are consistent across their notes. Preparation documentation
acknowledges clinical-cohort schema inspiration; this is not a claim of documented
clean-room authorship. The generator has no usable source-control history. A
limited comparison of 120 long template literals against the referenced schema
CSV found no complete normalized literal matches.

The public derivative removes the hardcoded hospital heading and adds a clear
synthetic notice. It excludes duplicated note text, local paths and
history/provenance objects from the raw labels and summaries.

## Known Annotation Limits

- Events are template-assigned, not clinician-adjudicated gold. Term/grade
  compatibility and clinical realism have not been independently certified.
- In the German source, 1,498 of 5,987 events have supplementary label evidence
  that the renderer omitted from the note. Main event spans are present. The
  public projection omits all upstream evidence snippets rather than presenting
  these absent spans as retrievable evidence.
- Events are sampled across encounters without a complete clinical trajectory
  model. German pre-treatment notes can print planned future therapy dates in
  their full history. Do not equate every mentioned date with a completed event.
- Full-cohort labels are note-level events, not typed gold answers for the demo
  feature lists. No patient-level answers are inferred from hidden source metadata.
- Shared templates and synthetic patient splits do not measure real-world clinical
  generalization. The English and German cohorts are not paired translations.

## Reproducibility And Release Controls

The reviewed-input fingerprints cover only projected notes, relative registries,
allowlisted labels and patient splits. Private source identifiers and raw metadata
are not hashed into these public fingerprints.

| Projected input | SHA-256 |
| --- | --- |
| English | `be2737652e94d01e37f6d0d4c318cb35900216fe3579b5674faedc5d1c7e044a` |
| German | `def4936409b0b7aaf74c4046b920797f915a22f67350d149f22c12ab2b2106da` |

`scripts/prepare_reviewed_cohorts.py` requires these exact inputs before adding
notices. `scripts/prepare_public_release.py --include-datasets` then checks pinned
output manifests and every payload file, rejecting changes, missing files or extra
files. Each dataset manifest records the review scope and text transformations.
Neither command commits, pushes, or changes repository visibility.

Reviewed source-code SHA-256 identifiers:

- Hybrid English builder: `9ad6dd22bb546ed556ac071d60ae3900b4a19c6903a19cf1c2f38b6532c15259`.
- Current CTCAE note generator: `7c5e4cb8962b946762843981f93d9146dfef4a41a00cf9f68f0e22f45a92cb8d`.
- Standalone German generator: `d21a0d5dba063b94e2be2e22a0641b0d5d5c9fd030c1e557f8415fb8f0faceaf`.

## Attribution And Rights Scope

The repository's requested license applies to the OncoRAG-authored synthetic-note
derivatives and release metadata. This technical review does not establish
ownership of every historical template or replace institutional review where
required. No original clinical records or restricted clinical source datasets are
granted redistribution rights here.

The English generation chain used [Synthea](https://github.com/synthetichealth/synthea),
whose software uses the [Apache License 2.0](https://github.com/synthetichealth/synthea/blob/master/LICENSE).
This does not make the separately rendered note templates Synthea note exports or
automatically assign them Synthea's license.

Event labels reference the National Cancer Institute's
[CTCAE resources](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events).
[CTCAE v6.0](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf)
incorporates MedDRA terminology. No complete terminology tables, instruments or
clinical source templates are bundled. NCI's
[reuse policy](https://www.cancer.gov/policies/copyright-reuse) distinguishes its own
material from separately protected third-party content. The repository's license
does not purport to override those rights.
