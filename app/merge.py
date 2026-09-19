"""Merge several named Intel HEX images into a single firmware image.

Every source image is parsed independently with the shared Intel HEX
parser. When any source fails validation its name and original parse
diagnostics are reported and no merge is attempted. Valid sources are
combined by absolute address:

* overlapping bytes with identical values merge cleanly, and every
  source contributing a byte is recorded;
* overlapping bytes with different values are reported as contiguous
  conflict ranges carrying each source's byte values, and no merged
  image is produced.

Entry addresses must either be absent from every source or agree on a
single non-null value; otherwise an ``entry_conflict`` is reported and
no merged image is produced either.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from .hex_parser import LineDiagnostic, ParseResult, Segment, parse_intel_hex


@dataclass
class SourceImage:
    """One named Intel HEX input image."""

    name: str
    hex_text: str


@dataclass
class InvalidSource:
    """A source whose HEX text failed validation."""

    name: str
    error_count: int
    warning_count: int
    diagnostics: list[LineDiagnostic]


@dataclass
class ConflictSource:
    """A source's bytes clipped to a byte-conflict range."""

    name: str
    segments: list[Segment]


@dataclass
class ByteConflict:
    """A maximal run of consecutive addresses where sources disagree."""

    start: int  # inclusive
    end: int  # exclusive
    sources: list[ConflictSource]


@dataclass
class SourceEntry:
    """A source's non-null entry address."""

    name: str
    entry_address: int


@dataclass
class EntryConflict:
    """Sources disagree on the entry address."""

    entries: list[SourceEntry]
    message: str


@dataclass
class MergedSegment:
    """A contiguous run of merged bytes and the sources behind it."""

    start: int
    data: bytes
    sources: list[str]  # contributing source names, in input order

    @property
    def end(self) -> int:
        """Exclusive end address."""
        return self.start + len(self.data)


@dataclass
class MergeResult:
    ok: bool
    image_count: int
    duplicate_names: list[str]
    invalid_sources: list[InvalidSource]
    entry_address: int | None
    entry_conflict: EntryConflict | None
    conflicts: list[ByteConflict]
    merged_segments: list[MergedSegment]
    total_bytes: int


def _fmt(addr: int) -> str:
    return f"0x{addr:04X}"


def merge_images(images: list[SourceImage]) -> MergeResult:
    """Merge ``images`` into one firmware image, or report why not.

    Duplicate source names are rejected up front. Each source is then
    parsed with the shared Intel HEX parser; any invalid source aborts
    the merge. Byte-level disagreements between valid sources are
    collected as contiguous conflict ranges and diverging entry
    addresses as an entry conflict; either one suppresses the merged
    image.
    """
    seen: set[str] = set()
    dupes: set[str] = set()
    for image in images:
        if image.name in seen:
            dupes.add(image.name)
        seen.add(image.name)
    if dupes:
        return MergeResult(
            ok=False,
            image_count=len(images),
            duplicate_names=sorted(dupes),
            invalid_sources=[],
            entry_address=None,
            entry_conflict=None,
            conflicts=[],
            merged_segments=[],
            total_bytes=0,
        )

    parsed: list[tuple[str, ParseResult]] = []
    invalid: list[InvalidSource] = []
    for image in images:
        result = parse_intel_hex(image.hex_text)
        if result.ok:
            parsed.append((image.name, result))
        else:
            invalid.append(
                InvalidSource(
                    name=image.name,
                    error_count=result.error_count,
                    warning_count=result.warning_count,
                    diagnostics=result.diagnostics,
                )
            )
    if invalid:
        return MergeResult(
            ok=False,
            image_count=len(images),
            duplicate_names=[],
            invalid_sources=invalid,
            entry_address=None,
            entry_conflict=None,
            conflicts=[],
            merged_segments=[],
            total_bytes=0,
        )

    order = [name for name, _ in parsed]
    source_maps: dict[str, dict[int, int]] = {}
    source_addrs: dict[str, list[int]] = {}
    for name, result in parsed:
        memory: dict[int, int] = {}
        for seg in result.segments:
            for i, value in enumerate(seg.data):
                memory[seg.start + i] = value
        source_maps[name] = memory
        source_addrs[name] = sorted(memory)

    all_addrs: set[int] = set()
    for memory in source_maps.values():
        all_addrs.update(memory)
    conflicted = sorted(
        addr
        for addr in all_addrs
        if len({m[addr] for m in source_maps.values() if addr in m}) > 1
    )
    conflicts = [
        ByteConflict(
            start=start,
            end=end,
            sources=[
                ConflictSource(
                    name=name,
                    segments=_clip_segments(source_maps[name], source_addrs[name], start, end),
                )
                for name in order
                if _covers(source_addrs[name], start, end)
            ],
        )
        for start, end in _contiguous_ranges(conflicted)
    ]

    entries = [
        SourceEntry(name=name, entry_address=result.entry_address)
        for name, result in parsed
        if result.entry_address is not None
    ]
    distinct = {e.entry_address for e in entries}
    entry_address: int | None = None
    entry_conflict: EntryConflict | None = None
    if len(distinct) > 1:
        detail = ", ".join(f"{e.name}={_fmt(e.entry_address)}" for e in entries)
        entry_conflict = EntryConflict(
            entries=entries,
            message=f"entry addresses disagree across sources: {detail}",
        )
    elif entries:
        entry_address = entries[0].entry_address

    merged: list[MergedSegment] = []
    if not conflicts and entry_conflict is None:
        union: dict[int, int] = {}
        contributors: dict[int, set[str]] = {}
        for name in order:
            for addr, value in source_maps[name].items():
                union[addr] = value
                contributors.setdefault(addr, set()).add(name)
        for start, end in _contiguous_ranges(sorted(union)):
            names: set[str] = set()
            blob = bytearray()
            for addr in range(start, end):
                blob.append(union[addr])
                names.update(contributors[addr])
            merged.append(
                MergedSegment(
                    start=start,
                    data=bytes(blob),
                    sources=[name for name in order if name in names],
                )
            )

    return MergeResult(
        ok=not conflicts and entry_conflict is None,
        image_count=len(images),
        duplicate_names=[],
        invalid_sources=[],
        entry_address=entry_address,
        entry_conflict=entry_conflict,
        conflicts=conflicts,
        merged_segments=merged,
        total_bytes=sum(len(s.data) for s in merged),
    )


def _contiguous_ranges(addresses: list[int]) -> list[tuple[int, int]]:
    """Group sorted, unique addresses into ``[start, end)`` runs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    prev = 0
    for addr in addresses:
        if start is None:
            start = addr
        elif addr != prev + 1:
            runs.append((start, prev + 1))
            start = addr
        prev = addr
    if start is not None:
        runs.append((start, prev + 1))
    return runs


def _covers(sorted_addrs: list[int], start: int, end: int) -> bool:
    """True when the sorted address list holds any address in ``[start, end)``."""
    return bisect_left(sorted_addrs, start) < bisect_left(sorted_addrs, end)


def _clip_segments(
    memory: dict[int, int],
    sorted_addrs: list[int],
    start: int,
    end: int,
) -> list[Segment]:
    """Cut a source's bytes down to its contiguous runs inside ``[start, end)``."""
    lo = bisect_left(sorted_addrs, start)
    hi = bisect_left(sorted_addrs, end)
    return [
        Segment(start=run_start, data=bytes(memory[a] for a in range(run_start, run_end)))
        for run_start, run_end in _contiguous_ranges(sorted_addrs[lo:hi])
    ]
