import spacy

SCISPACY_MODELS = [
    "en_ner_bionlp13cg_md",
    "en_ner_bc5cdr_md",
    "en_ner_chemprot_md",
    "en_ner_craft_md",
    "en_ner_jnlpba_md",
    "en_core_sci_sm",
    "en_core_sci_md",
    "en_core_sci_lg",
    "en_core_sci_scibert",
]

for name in SCISPACY_MODELS:
    try:
        nlp = spacy.load(name)          # auto-downloads if not installed
    except Exception as exc:
        print(f"{name}: not available ({exc})")
        continue

    ner = nlp.get_pipe("ner")
    print(f"{name}: {len(ner.labels)} labels")
    for label in sorted(ner.labels):
        print(f"   {label}")
