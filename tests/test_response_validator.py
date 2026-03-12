"""Tests for LLM response validation and pathology detection."""

from __future__ import annotations

from library.harness.response_validator import is_response_pathological


class TestResponseValidator:
    def test_normal_response_passes(self):
        """A well-formed response should pass validation."""
        text = """This is a thoughtful response.

It has multiple paragraphs and natural structure.

Each paragraph says something different and meaningful."""
        is_bad, reason = is_response_pathological(text)
        assert not is_bad
        assert reason == ""

    def test_repeating_paragraphs_rejected(self):
        """Responses with identical paragraphs should be rejected."""
        text = """This is a paragraph that repeats.

Some other text in between.

This is a paragraph that repeats."""
        is_bad, reason = is_response_pathological(text)
        assert is_bad
        assert "paragraph" in reason.lower()

    def test_repeating_sentences_rejected(self):
        """Responses with sentences appearing multiple times should be rejected."""
        text = """I think this is important. It really is important.
Let me reiterate. I think this is important.
This is critical."""
        is_bad, reason = is_response_pathological(text)
        assert is_bad
        assert "sentence" in reason.lower() or "appears" in reason.lower()

    def test_excessive_length_without_breaks_rejected(self):
        """Long text without enough paragraph breaks should be rejected."""
        # ~1500 words in a single paragraph
        long_text = " ".join(["word"] * 1500)
        is_bad, reason = is_response_pathological(long_text)
        assert is_bad
        assert "paragraphs" in reason.lower() or "words" in reason.lower()

    def test_500_words_fewer_than_3_paragraphs_rejected(self):
        """500+ words split into fewer than 3 paragraphs should be rejected."""
        # ~600 words in 2 paragraphs
        text = " ".join(["word"] * 300) + "\n\n" + " ".join(["word"] * 300)
        is_bad, reason = is_response_pathological(text)
        assert is_bad

    def test_500_words_with_3_paragraphs_passes(self):
        """500+ words split into 3+ paragraphs should pass."""
        # ~600 words in 3 paragraphs (different content)
        para1 = " ".join(["word"] * 200)
        para2 = " ".join(["phrase"] * 200)
        para3 = " ".join(["text"] * 200)
        text = f"{para1}\n\n{para2}\n\n{para3}"
        is_bad, reason = is_response_pathological(text)
        assert not is_bad

    def test_1000_words_fewer_than_5_paragraphs_rejected(self):
        """1000+ words with fewer than 5 paragraphs should be rejected."""
        # ~1200 words in 4 paragraphs
        para = " ".join(["word"] * 300)
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        is_bad, reason = is_response_pathological(text)
        assert is_bad

    def test_1000_words_with_5_paragraphs_passes(self):
        """1000+ words with 5+ paragraphs should pass."""
        # ~1000 words in 5 paragraphs (different content)
        paras = [
            " ".join([f"word{i}"] * 200)
            for i in range(5)
        ]
        text = "\n\n".join(paras)
        is_bad, reason = is_response_pathological(text)
        assert not is_bad

    def test_empty_response_passes(self):
        """Empty or whitespace-only responses should pass (other code handles them)."""
        is_bad, reason = is_response_pathological("")
        assert not is_bad

        is_bad, reason = is_response_pathological("   \n\n  ")
        assert not is_bad

    def test_short_text_passes(self):
        """Short text (< 500 words) with few paragraphs should pass."""
        text = "This is a short response.\n\nWith two paragraphs."
        is_bad, reason = is_response_pathological(text)
        assert not is_bad
