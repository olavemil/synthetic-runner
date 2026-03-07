"""Tests for symbiosis.harness.sanitize."""

from symbiosis.harness.sanitize import strip_think_blocks


class TestStripThinkBlocks:
    def test_paired_tags(self):
        text = "<think>private reasoning</think>\nVisible."
        assert strip_think_blocks(text) == "Visible."

    def test_fenced_blocks(self):
        text = "```thinking\ninternal notes\n```\nVisible."
        assert strip_think_blocks(text) == "Visible."

    def test_bare_open_tag(self):
        text = "<think>line without close tag\nVisible answer."
        result = strip_think_blocks(text)
        assert "Visible answer." in result
        assert "<think>" not in result

    def test_mixed_content(self):
        text = (
            "<think>chain of thought</think>\n"
            "```reasoning\nstep by step\n```\n"
            "<analysis>deep dive</analysis>\n"
            "The actual response."
        )
        result = strip_think_blocks(text)
        assert "chain of thought" not in result
        assert "step by step" not in result
        assert "deep dive" not in result
        assert result == "The actual response."

    def test_empty_input(self):
        assert strip_think_blocks("") == ""
        assert strip_think_blocks(None) == ""

    def test_no_think_blocks(self):
        text = "Just a normal response."
        assert strip_think_blocks(text) == "Just a normal response."

    def test_single_closing_tag(self):
        text = "</think>\nVisible."
        result = strip_think_blocks(text)
        assert "</think>" not in result
        assert "Visible." in result

    def test_thinking_variant(self):
        text = "<thinking>internal</thinking>\nOutput."
        assert strip_think_blocks(text) == "Output."

    def test_reasoning_variant(self):
        text = "<reasoning>process</reasoning>\nResult."
        assert strip_think_blocks(text) == "Result."
