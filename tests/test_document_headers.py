"""German letter imports use generic document headings, not named institutions."""

import pytest

from oncoraggraph.graph.graph_builder import _clean_ricci_document, _split_ricci_style_notes


@pytest.mark.parametrize("heading", [
    "Universit\u00e4tsklinikum Beispielstadt",
    "Universitaetsklinikum Beispielstadt",
    "Universitatsklinikum Beispielstadt",
    "Klinikum Beispielstadt",
    "Krankenhaus Beispielstadt",
    "| UNIVERSITAETSKLINIKUM BEISPIELSTADT",
])
def test_generic_german_headings_separate_letters(heading):
    first = "Patient SYN-DEMO-001\nDiagnose: Glioblastom."
    second = "Patient SYN-DEMO-002\nTherapie: Strahlentherapie."
    documents = f"{heading}\n{first}\n{heading}\n{second}\n"
    assert _split_ricci_style_notes(documents) == [first, second]


def test_generic_administrative_headers_are_removed_without_content_loss():
    note = (
        "Klinikum Beispielstadt\nAnschrift: Synthetische Beispieladresse\n"
        "Postfach: Beispiel\n\nBefund: Keine neuen Beschwerden.\n"
    )
    assert _clean_ricci_document(note) == "Befund: Keine neuen Beschwerden."


def test_clinical_mentions_do_not_split_documents():
    note = "Patient SYN-DEMO-001\nEine Vorstellung im Klinikum wurde geplant.\nTherapie: unveraendert."
    assert _split_ricci_style_notes(note) == [note]


@pytest.mark.parametrize("note", [
    "Klinik: Keine neuen Beschwerden.",
    "Krankenhausaufenthalt: nicht erforderlich.",
])
def test_clinical_findings_are_not_administrative_headers(note):
    assert _clean_ricci_document(note) == note
