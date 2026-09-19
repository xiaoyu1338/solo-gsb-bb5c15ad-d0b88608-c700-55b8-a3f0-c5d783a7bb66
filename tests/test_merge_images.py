"""Tests for multi-image merging and the /api/v1/merge-images endpoint."""

import base64

from fastapi.testclient import TestClient

from app.main import app
from app.merge import SourceImage, merge_images

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def hex_text(*lines: str) -> str:
    return "\n".join([*lines, EOF])


def merge(*images: tuple[str, str]):
    return merge_images([SourceImage(name=n, hex_text=t) for n, t in images])


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_disjoint_images_merge_into_address_ordered_segments():
    result = merge(
        ("boot", hex_text(data_rec(0x0000, b"\x01\x02"))),
        ("app", hex_text(data_rec(0x0010, b"\x03\x04"))),
    )
    assert result.ok
    assert result.image_count == 2
    assert len(result.merged_segments) == 2
    first, second = result.merged_segments
    assert (first.start, first.data, first.sources) == (0x0000, b"\x01\x02", ["boot"])
    assert (second.start, second.data, second.sources) == (0x0010, b"\x03\x04", ["app"])
    assert result.total_bytes == 4
    assert result.conflicts == []
    assert result.entry_conflict is None


def test_adjacent_images_join_one_segment_with_both_sources():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01\x02"))),
        ("b", hex_text(data_rec(0x0002, b"\x03\x04"))),
    )
    assert result.ok
    assert len(result.merged_segments) == 1
    seg = result.merged_segments[0]
    assert (seg.start, seg.end, seg.data) == (0x0000, 0x0004, b"\x01\x02\x03\x04")
    assert seg.sources == ["a", "b"]


def test_identical_overlap_merges_and_records_all_sources():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01\x02\x03"))),
        ("b", hex_text(data_rec(0x0001, b"\x02\x03\x04"))),
    )
    assert result.ok
    assert len(result.merged_segments) == 1
    seg = result.merged_segments[0]
    assert (seg.start, seg.data) == (0x0000, b"\x01\x02\x03\x04")
    assert seg.sources == ["a", "b"]  # both contribute to the segment


def test_absolute_addresses_from_extended_records_are_merged():
    text_a = "\n".join([rec(0x04, 0, b"\x08\x00"), data_rec(0x0000, b"\xAA"), EOF])
    text_b = "\n".join([rec(0x04, 0, b"\x08\x00"), data_rec(0x0001, b"\xBB"), EOF])
    result = merge(("a", text_a), ("b", text_b))
    assert result.ok
    assert len(result.merged_segments) == 1
    seg = result.merged_segments[0]
    assert (seg.start, seg.data) == (0x08000000, b"\xAA\xBB")


def test_single_image_merges_to_itself():
    result = merge(("only", hex_text(data_rec(0x0000, b"\x01"))))
    assert result.ok
    assert result.merged_segments[0].sources == ["only"]


# ---------------------------------------------------------------------------
# byte conflicts
# ---------------------------------------------------------------------------

def test_differing_overlap_reports_contiguous_conflict_ranges():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01\x02\x03\x04"))),
        ("b", hex_text(data_rec(0x0001, b"\x02\xFF\xFE\x04"))),
    )
    assert not result.ok
    assert result.merged_segments == []  # no merged image on conflict
    assert result.total_bytes == 0
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert (conflict.start, conflict.end) == (0x0002, 0x0004)
    values = {s.name: s.segments[0].data for s in conflict.sources}
    assert values == {"a": b"\x03\x04", "b": b"\xFF\xFE"}


def test_separate_conflicts_yield_separate_ranges():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01\x02\x03\x04\x05"))),
        ("b", hex_text(data_rec(0x0000, b"\x09\x02\x03\x04\x08"))),
    )
    assert not result.ok
    assert [(c.start, c.end) for c in result.conflicts] == [(0x0000, 0x0001), (0x0004, 0x0005)]


def test_conflict_lists_every_source_covering_the_range():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01\x02"))),
        ("b", hex_text(data_rec(0x0000, b"\x09\x02"))),
        ("c", hex_text(data_rec(0x0000, b"\x01"))),  # agrees with "a", still a source
    )
    assert not result.ok
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert (conflict.start, conflict.end) == (0x0000, 0x0001)
    assert [s.name for s in conflict.sources] == ["a", "b", "c"]
    values = {s.name: s.segments[0].data for s in conflict.sources}
    assert values == {"a": b"\x01", "b": b"\x09", "c": b"\x01"}


def test_entry_address_still_reported_when_bytes_conflict():
    entry = rec(0x05, 0, b"\x08\x00\x01\xC1")
    result = merge(
        ("a", "\n".join([data_rec(0x0000, b"\x01"), entry, EOF])),
        ("b", "\n".join([data_rec(0x0000, b"\x02"), entry, EOF])),
    )
    assert not result.ok
    assert result.entry_conflict is None
    assert result.entry_address == 0x080001C1  # sources agree on the entry


# ---------------------------------------------------------------------------
# entry address rules
# ---------------------------------------------------------------------------

def test_entry_absent_everywhere_is_none():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01"))),
        ("b", hex_text(data_rec(0x0010, b"\x02"))),
    )
    assert result.ok
    assert result.entry_address is None
    assert result.entry_conflict is None


def test_equal_non_null_entries_merge():
    entry = rec(0x05, 0, b"\x08\x00\x01\xC1")
    result = merge(
        ("a", "\n".join([data_rec(0x0000, b"\x01"), entry, EOF])),
        ("b", "\n".join([data_rec(0x0010, b"\x02"), entry, EOF])),
    )
    assert result.ok
    assert result.entry_address == 0x080001C1


def test_missing_and_present_entry_merge():
    entry = rec(0x05, 0, b"\x08\x00\x01\xC1")
    result = merge(
        ("a", "\n".join([data_rec(0x0000, b"\x01"), entry, EOF])),
        ("b", hex_text(data_rec(0x0010, b"\x02"))),
    )
    assert result.ok
    assert result.entry_address == 0x080001C1


def test_differing_entries_report_entry_conflict_and_suppress_merge():
    result = merge(
        ("a", "\n".join([data_rec(0x0000, b"\x01"), rec(0x05, 0, b"\x08\x00\x01\xC1"), EOF])),
        ("b", "\n".join([data_rec(0x0010, b"\x02"), rec(0x05, 0, b"\x08\x00\x02\x00"), EOF])),
    )
    assert not result.ok
    assert result.merged_segments == []
    assert result.entry_address is None
    conflict = result.entry_conflict
    assert conflict is not None
    assert {e.name: e.entry_address for e in conflict.entries} == {
        "a": 0x080001C1,
        "b": 0x08000200,
    }
    assert "a" in conflict.message and "b" in conflict.message


# ---------------------------------------------------------------------------
# invalid sources and duplicate names
# ---------------------------------------------------------------------------

def test_invalid_source_aborts_merge_and_keeps_original_diagnostics():
    bad = data_rec(0x0000, b"\x11\x22")
    bad = bad[:-2] + "00"  # corrupt checksum
    result = merge(
        ("good", hex_text(data_rec(0x0000, b"\x01"))),
        ("bad", "\n".join([bad, EOF])),
    )
    assert not result.ok
    assert result.merged_segments == []
    assert len(result.invalid_sources) == 1
    source = result.invalid_sources[0]
    assert source.name == "bad"
    assert source.error_count == 1
    assert any("checksum" in d.message and d.line == 1 for d in source.diagnostics)


def test_duplicate_names_rejected_before_parsing():
    result = merge(
        ("a", hex_text(data_rec(0x0000, b"\x01"))),
        ("a", "not even hex"),
    )
    assert not result.ok
    assert result.duplicate_names == ["a"]
    assert result.invalid_sources == []  # duplicate check wins, nothing parsed
    assert result.merged_segments == []


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_merge_images_roundtrip():
    resp = client.post(
        "/api/v1/merge-images",
        json={
            "images": [
                {
                    "name": "boot",
                    "hex_text": "\n".join([
                        rec(0x04, 0, b"\x08\x00"),
                        data_rec(0x0000, bytes(range(16))),
                        rec(0x05, 0, b"\x08\x00\x00\x09"),
                        EOF,
                    ]),
                },
                {
                    "name": "app",
                    "hex_text": "\n".join([
                        rec(0x04, 0, b"\x08\x00"),
                        data_rec(0x0010, bytes(range(16, 32))),
                        rec(0x05, 0, b"\x08\x00\x00\x09"),
                        EOF,
                    ]),
                },
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["image_count"] == 2
    assert body["entry_address"] == 0x08000009
    assert body["entry_conflict"] is None
    assert body["conflicts"] == []
    assert body["invalid_sources"] == []
    assert body["duplicate_names"] == []
    assert body["segment_count"] == 1
    seg = body["merged_segments"][0]
    assert seg["start"] == 0x08000000
    assert seg["end"] == 0x08000020
    assert seg["size"] == 32
    assert base64.b64decode(seg["data_base64"]) == bytes(range(32))
    assert seg["sources"] == ["boot", "app"]
    assert body["total_bytes"] == 32


def test_api_reports_invalid_source_with_200_and_ok_false():
    resp = client.post(
        "/api/v1/merge-images",
        json={
            "images": [
                {"name": "good", "hex_text": hex_text(data_rec(0, b"\x01"))},
                {"name": "bad", "hex_text": ":00000001FE\n"},  # bad checksum
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["merged_segments"] == []
    assert [s["name"] for s in body["invalid_sources"]] == ["bad"]
    assert body["invalid_sources"][0]["error_count"] >= 1
    assert any(
        "checksum" in d["message"] for d in body["invalid_sources"][0]["diagnostics"]
    )


def test_api_reports_byte_conflict_without_merged_segments():
    resp = client.post(
        "/api/v1/merge-images",
        json={
            "images": [
                {"name": "a", "hex_text": hex_text(data_rec(0x0000, b"\x01\x02"))},
                {"name": "b", "hex_text": hex_text(data_rec(0x0001, b"\xFF"))},
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["merged_segments"] == []
    assert body["segment_count"] == 0
    assert len(body["conflicts"]) == 1
    conflict = body["conflicts"][0]
    assert (conflict["start"], conflict["end"], conflict["size"]) == (1, 2, 1)
    values = {
        s["name"]: base64.b64decode(s["segments"][0]["data_base64"])
        for s in conflict["sources"]
    }
    assert values == {"a": b"\x02", "b": b"\xFF"}


def test_api_reports_entry_conflict():
    resp = client.post(
        "/api/v1/merge-images",
        json={
            "images": [
                {
                    "name": "a",
                    "hex_text": "\n".join(
                        [data_rec(0, b"\x01"), rec(0x05, 0, b"\x00\x00\x00\x01"), EOF]
                    ),
                },
                {
                    "name": "b",
                    "hex_text": "\n".join(
                        [data_rec(0x10, b"\x02"), rec(0x05, 0, b"\x00\x00\x00\x02"), EOF]
                    ),
                },
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["merged_segments"] == []
    assert body["entry_address"] is None
    entries = {e["name"]: e["entry_address"] for e in body["entry_conflict"]["entries"]}
    assert entries == {"a": 1, "b": 2}


def test_api_rejects_duplicate_names():
    resp = client.post(
        "/api/v1/merge-images",
        json={
            "images": [
                {"name": "a", "hex_text": hex_text(data_rec(0, b"\x01"))},
                {"name": "a", "hex_text": hex_text(data_rec(1, b"\x02"))},
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["duplicate_names"] == ["a"]
    assert body["merged_segments"] == []


def test_api_requires_at_least_one_image():
    resp = client.post("/api/v1/merge-images", json={"images": []})
    assert resp.status_code == 422
