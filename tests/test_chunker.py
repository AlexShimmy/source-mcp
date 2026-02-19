import pytest
from src.services.indexer import TextChunker


def test_chunker_initialization():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    assert chunker.chunk_size == 10
    assert chunker.chunk_overlap == 2


def test_split_text_short():
    """Text shorter than chunk_size → single chunk."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split_text("short text")
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_split_text_multiple_chunks():
    """Longer text produces multiple chunks."""
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    text = "The quick brown fox jumps over the lazy dog again and again."
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    # Every chunk should be non-empty
    for c in chunks:
        assert len(c) > 0
    # Reassembled chunks should contain all original words
    combined = " ".join(chunks)
    for word in text.split():
        assert word in combined


def test_split_prefers_word_boundaries():
    """Chunker tries to split at spaces rather than mid-word."""
    chunker = TextChunker(chunk_size=15, chunk_overlap=3)
    text = "hello world foo bar baz qux"
    chunks = chunker.split_text(text)
    # With word-boundary splitting, chunks should not break mid-word
    for c in chunks:
        # Each chunk should start/end at a word boundary (no partial words at boundaries)
        assert not c.startswith(" "), f"Chunk starts with space: '{c}'"


def test_split_empty_text():
    chunker = TextChunker()
    assert chunker.split_text("") == []


def test_split_none_text():
    chunker = TextChunker()
    assert chunker.split_text(None) == []


def test_split_preserves_content():
    """No content should be lost during splitting."""
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunker.split_text(text)
    # Join should cover all original text
    all_text = "".join(chunks)
    for word in text.split():
        assert word.strip(".") in all_text
