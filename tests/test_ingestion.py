import csv
import json
from types import SimpleNamespace

import pytest

from oncoraggraph.ingestion import NoteRecord, group_notes_by_patient, load_notes
from oncoraggraph.graph import graph_builder


def write_note(root, relative, text="Fatigue is documented."):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_registry(path, rows):
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif path.suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows), encoding="utf-8")


def test_folder_keeps_same_date_different_reports_and_suffixes(tmp_path):
    write_note(tmp_path, "patient1/oncology/2025-01-02.txt")
    write_note(tmp_path, "patient1/pathology/2025-01-02.txt")
    write_note(tmp_path, "patient1/oncology/2025-01-02__amended.txt")
    write_note(tmp_path, "patient1/oncology/2025-01-01.txt")

    notes = load_notes(notes_root=tmp_path, default_language="english")

    assert len(notes) == 4
    assert len({note.note_id for note in notes}) == 4
    assert notes[0].date == "2025-01-01"
    assert {note.report_type for note in notes} == {"oncology", "pathology"}
    assert {note.language for note in notes} == {"en"}
    assert list(group_notes_by_patient(notes)) == ["patient1"]


@pytest.mark.parametrize("extension", [".csv", ".json", ".jsonl"])
def test_registry_matches_folder_and_resolves_paths_from_registry(tmp_path, extension, monkeypatch):
    root = tmp_path / "notes"
    write_note(root, "0001/oncology/2025-01-02.txt")
    folder_records = load_notes(notes_root=root, default_language="en")
    registry = tmp_path / f"registry{extension}"
    write_registry(registry, [{
        "patient_id": "0001", "report_type": "oncology", "date": "2025-01-02",
        "path": "notes/0001/oncology/2025-01-02.txt", "language": "english",
    }])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_notes(registry_path=registry) == folder_records


def test_json_object_registry_keeps_explicit_metadata_and_note_languages(tmp_path):
    write_note(tmp_path, "english.txt", "Date is 01/01/2001. Fatigue is documented.")
    write_note(tmp_path, "german.txt", "Fatigue ist dokumentiert.")
    rows = [{
        "patient_id": "p1", "note_id": "n1", "report_type": "oncology",
        "date": "2025-01-02", "path": "english.txt", "language": "en",
    }, {
        "patient_id": "p1", "note_id": "n2", "report_type": "pathology",
        "date": "2025-01-03", "path": "german.txt", "language": "german",
    }]
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"notes": rows}), encoding="utf-8")

    notes = load_notes(registry_path=registry)

    assert [note.note_id for note in notes] == ["n1", "n2"]
    assert [note.date for note in notes] == ["2025-01-02", "2025-01-03"]
    assert [note.language for note in notes] == ["en", "de"]
    assert notes[1].text == "Fatigue ist dokumentiert."


@pytest.mark.parametrize("relative", [
    "p1/2025-01-01.txt", "p1/oncology/subdir/2025-01-01.txt",
    "p1/oncology/2025-1-01.txt", "p1/oncology/2025-01-01__.txt",
    "p1/oncology/2025-02-29.txt", "p1/oncology/2025-13-01.txt",
])
def test_folder_rejects_invalid_layout_or_date(tmp_path, relative):
    write_note(tmp_path, relative)
    with pytest.raises(ValueError):
        load_notes(notes_root=tmp_path)


def test_no_source_or_two_sources_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_notes()
    with pytest.raises(ValueError, match="exactly one"):
        load_notes(notes_root=tmp_path, registry_path=tmp_path / "registry.csv")


def test_empty_and_blank_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="No clinical notes"):
        load_notes(notes_root=tmp_path)
    write_note(tmp_path, "p1/oncology/2025-01-01.txt", " \n\t")
    with pytest.raises(ValueError, match="blank text"):
        load_notes(notes_root=tmp_path)


@pytest.mark.parametrize("field,value", [
    ("date", "2025-02-29"), ("patient_id", 1), ("report_type", ""),
    ("path", "absent.txt"), ("language", "invalid-language"),
    ("patient_id", "../p1"), ("report_type", "../notes"),
])
def test_registry_rejects_invalid_records(tmp_path, field, value):
    write_note(tmp_path, "note.txt")
    row = {"patient_id": "p1", "note_id": "n1", "report_type": "oncology",
           "date": "2025-01-02", "path": "note.txt", "language": "en"}
    row[field] = value
    registry = tmp_path / "registry.json"
    write_registry(registry, [row])
    with pytest.raises(ValueError):
        load_notes(registry_path=registry)


def test_registry_rejects_duplicate_note_ids_and_source_files(tmp_path):
    write_note(tmp_path, "note.txt")
    write_note(tmp_path, "second.txt")
    row = {"patient_id": "p1", "note_id": "n1", "report_type": "oncology",
           "date": "2025-01-02", "path": "note.txt"}
    registry = tmp_path / "registry.json"
    write_registry(registry, [row, {**row, "path": "second.txt"}])
    with pytest.raises(ValueError, match="Duplicate note_id"):
        load_notes(registry_path=registry)
    write_registry(registry, [row, {**row, "note_id": "n2"}])
    with pytest.raises(ValueError, match="Duplicate note source"):
        load_notes(registry_path=registry)


@pytest.mark.parametrize("content", [
    "patient_id,report_type,date,path,path\np1,oncology,2025-01-01,note.txt,note.txt\n",
    "patient_id,report_type,date,path\np1,oncology,2025-01-01,note.txt,extra\n",
    "patient_id,date,path\np1,2025-01-01,note.txt\n",
])
def test_registry_rejects_malformed_csv(tmp_path, content):
    registry = tmp_path / "registry.csv"
    registry.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_notes(registry_path=registry)


@pytest.fixture
def fake_graph_models(monkeypatch):
    calls = []

    def entities(text, models, filters, deduplication):
        calls.append((text, models, filters, deduplication))
        return [{"text": "Fatigue", "label": "DISEASE"}]

    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", entities)
    monkeypatch.setattr(graph_builder, "get_scispacy_model", lambda name: (
        lambda text: SimpleNamespace(sents=[SimpleNamespace(text=text)])
    ))
    return calls


def test_patient_graph_keeps_authoritative_metadata_and_distinct_sentence_ids(tmp_path, fake_graph_models):
    path1 = write_note(tmp_path, "p1/oncology/2025-01-02.txt", "Date is 01/01/2001. Fatigue is documented.")
    path2 = write_note(tmp_path, "p1/pathology/2025-01-02.txt", "Fatigue ist dokumentiert.")
    notes = [
        NoteRecord("p1", "oncology/n1", "oncology", "2025-01-02", path1, path1.read_text(), "en"),
        NoteRecord("p1", "pathology/n1", "pathology", "2025-01-02", path2, path2.read_text(), "de"),
    ]

    graph = graph_builder.build_patient_graph(notes, model_configs=[{"name": "test-model"}])

    note_nodes = {node: attrs for node, attrs in graph.nodes(data=True) if attrs["label"] == "Note"}
    assert len(note_nodes) == 2
    assert {attrs["note_date"] for attrs in note_nodes.values()} == {"2025-01-02"}
    assert {attrs["report_type"] for attrs in note_nodes.values()} == {"oncology", "pathology"}
    assert {attrs["language"] for attrs in note_nodes.values()} == {"en", "de"}
    assert {attrs["note_path"] for attrs in note_nodes.values()} == {str(path1), str(path2)}
    assert {attrs["text"] for attrs in note_nodes.values()} == {note.text for note in notes}
    assert "01/01/2001" not in graph
    mentions = [attrs for _, _, attrs in graph.edges(data=True) if attrs.get("relation") == "MENTIONS"]
    assert {attrs["source_note_id"] for attrs in mentions} == set(note_nodes)
    assert len({attrs["source_sentence_id"] for attrs in mentions}) == 2
    assert graph.graph["languages"] == ["de", "en"]
    assert graph.graph["note_count"] == 2
    assert len(fake_graph_models) == 2


def test_graph_keeps_all_report_sentences_when_ner_finds_no_entities(tmp_path, monkeypatch):
    import spacy

    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    monkeypatch.setattr(graph_builder, "get_scispacy_model", lambda name: nlp)
    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", lambda *args: [])
    text = "Alter: 57 Jahre. Gewicht: 81 kg. Termin am 02.01.2025."
    note = NoteRecord("p1", "n1", "oncology", "2025-01-02", tmp_path / "note.txt", text, "de")

    graph = graph_builder.build_patient_graph([note], model_configs=[{"name": "test-model"}])

    sentences = {node: attrs for node, attrs in graph.nodes(data=True) if attrs["label"] == "Sentence"}
    assert len(sentences) == 3
    assert [attrs["original_text"] for attrs in sentences.values()] == [sent.text for sent in nlp(text).sents]
    assert {attrs["source_model"] for attrs in sentences.values()} == {"report_sentence"}
    assert {attrs["note_date"] for attrs in sentences.values()} == {"2025-01-02"}
    assert {attrs["language"] for attrs in sentences.values()} == {"de"}
    assert {attrs["report_type"] for attrs in sentences.values()} == {"oncology"}
    edges = [attrs for _, _, attrs in graph.edges(data=True) if attrs["relation"] == "CONTAINS_SENTENCE"]
    assert {attrs["source_sentence_id"] for attrs in edges} == set(sentences)
    assert graph.graph["includes_report_sentences"] is True

    graph_without_sentences = graph_builder.build_patient_graph(
        [note], model_configs=[{"name": "test-model"}], include_report_sentences=False,
    )
    assert all(attrs["label"] != "Sentence" for _, attrs in graph_without_sentences.nodes(data=True))


def test_graph_note_ids_are_patient_scoped(tmp_path, fake_graph_models):
    graphs = []
    for patient in ["p1", "p2"]:
        path = write_note(tmp_path, f"{patient}.txt")
        note = NoteRecord(patient, "n1", "oncology", "2025-01-02", path, path.read_text())
        graphs.append(graph_builder.build_patient_graph([note], model_configs=[{"name": "test-model"}]))
    ids = [{node for node, attrs in graph.nodes(data=True) if attrs["label"] == "Note"} for graph in graphs]
    assert ids[0].isdisjoint(ids[1])


def test_graph_skips_blank_spacy_spans_without_renumbering_sentences(tmp_path, monkeypatch):
    import spacy
    from spacy.tokens import Doc

    doc = Doc(spacy.blank("xx").vocab,
              words=["\n", "Weight", "is", "70", "kg", ".", "\n"],
              spaces=[False, True, True, True, False, False, False],
              sent_starts=[True, True, False, False, False, False, True])
    assert [sentence.text for sentence in doc.sents] == ["\n", "Weight is 70 kg.", "\n"]
    monkeypatch.setattr(graph_builder, "get_scispacy_model", lambda name: lambda text: doc)
    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", lambda *args: [])
    note = NoteRecord("p1", "n1", "oncology", "2025-01-02", tmp_path / "note.txt", doc.text, "en")

    graph = graph_builder.build_patient_graph([note], model_configs=[{"name": "test-model"}])

    sentences = {node: attrs for node, attrs in graph.nodes(data=True) if attrs["label"] == "Sentence"}
    assert list(sentences) == ["note:p1:n1_sent_1"]
    assert sentences["note:p1:n1_sent_1"]["original_text"] == "Weight is 70 kg."
    assert graph.edges["note:p1:n1", "note:p1:n1_sent_1"]["source_sentence_index"] == 1


def test_graph_rejects_zero_or_multiple_patients_before_model_calls(tmp_path, fake_graph_models):
    with pytest.raises(ValueError, match="exactly one patient"):
        graph_builder.build_patient_graph([], model_configs=[{"name": "test-model"}])
    notes = [NoteRecord(pid, "n1", "oncology", "2025-01-02", tmp_path / f"{pid}.txt", "Fatigue")
             for pid in ["p1", "p2"]]
    with pytest.raises(ValueError, match="exactly one patient"):
        graph_builder.build_patient_graph(notes, model_configs=[{"name": "test-model"}])
    assert fake_graph_models == []


def test_graph_propagates_extraction_failures(tmp_path, fake_graph_models, monkeypatch):
    def fail(*args):
        raise RuntimeError("entity extraction failed")

    monkeypatch.setattr(graph_builder, "extract_and_deduplicate_entities", fail)
    note = NoteRecord("p1", "n1", "oncology", "2025-01-02", tmp_path / "note.txt", "Fatigue")
    with pytest.raises(RuntimeError, match="entity extraction failed"):
        graph_builder.build_patient_graph([note], model_configs=[{"name": "test-model"}])


def test_legacy_graph_call_still_supports_body_dates(fake_graph_models):
    graph = graph_builder.process_notes_to_graph(
        ["Date is 01/02/2025. Fatigue is documented."], "p1", "notes.txt",
        [{"name": "test-model"}], {}, {},
    )
    assert graph.nodes["notes.txt_note_0"]["note_date"] == "01/02/2025"
    assert all(attrs["label"] != "Sentence" for _, attrs in graph.nodes(data=True))
