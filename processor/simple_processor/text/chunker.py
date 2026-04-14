"""Recursive character-based text splitter.

Splits text by a prioritized list of separators (paragraphs → lines →
sentences → words → hard split), so chunk boundaries land on semantic
breaks when possible. Preserves character offsets into the original text
so callers can recover surrounding context at retrieval time.
"""

from dataclasses import dataclass

DEFAULT_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",  # sentinel: hard split by size when nothing else fits
]


@dataclass
class RawChunk:
    """A raw (pre-embedding) chunk with its position in the source text."""

    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    *,
    size: int = 800,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[RawChunk]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Args:
        text: Input text.
        size: Max chunk length in characters.
        overlap: Number of characters the next chunk starts before the previous
            chunk ends. Must be ``0 <= overlap < size``.
        separators: Ordered list of separators to try, from most to least
            preferred. The empty string ``""`` acts as a sentinel meaning
            "fall back to hard splitting by size".

    Returns:
        List of ``RawChunk`` in document order.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    if not text:
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _split(text, 0, len(text), size, seps)
    return _merge(text, pieces, size, overlap)


def _split(
    text: str, lo: int, hi: int, size: int, separators: list[str],
) -> list[tuple[int, int]]:
    """Recursively split the range ``[lo, hi)`` into (start, end) pieces
    each of length <= size.
    """
    if hi - lo <= size:
        return [(lo, hi)] if hi > lo else []

    for idx, sep in enumerate(separators):
        if sep == "":
            return _hard_split(lo, hi, size)
        if text.find(sep, lo, hi) != -1:
            return _split_by_sep(text, lo, hi, sep, size, separators[idx + 1:])

    return _hard_split(lo, hi, size)


def _split_by_sep(
    text: str,
    lo: int,
    hi: int,
    sep: str,
    size: int,
    sub_separators: list[str],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = lo
    sl = len(sep)
    while i < hi:
        j = text.find(sep, i, hi)
        if j == -1:
            piece_end = hi
            next_i = hi
        else:
            piece_end = j + sl  # keep separator attached to the preceding piece
            next_i = j + sl

        if piece_end - i > size:
            result.extend(_split(text, i, piece_end, size, sub_separators))
        elif piece_end > i:
            result.append((i, piece_end))
        i = next_i
    return result


def _hard_split(lo: int, hi: int, size: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = lo
    while i < hi:
        j = min(i + size, hi)
        result.append((i, j))
        i = j
    return result


def _merge(
    text: str,
    pieces: list[tuple[int, int]],
    size: int,
    overlap: int,
) -> list[RawChunk]:
    """Greedily pack consecutive pieces into chunks of length <= size.

    When a new chunk starts, its ``start`` is pulled back by up to ``overlap``
    characters into the tail of the previous chunk.
    """
    if not pieces:
        return []

    chunks: list[tuple[int, int]] = []
    cur_s, cur_e = pieces[0]
    for ps, pe in pieces[1:]:
        if pe - cur_s <= size:
            cur_e = pe
        else:
            chunks.append((cur_s, cur_e))
            # new chunk: overlap into the tail of the one we just closed
            new_s = max(cur_e - overlap, 0) if overlap > 0 else ps
            new_s = min(new_s, ps)
            cur_s, cur_e = new_s, pe
    chunks.append((cur_s, cur_e))

    return [RawChunk(text=text[s:e], start=s, end=e) for s, e in chunks]
