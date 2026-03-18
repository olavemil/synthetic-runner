"""Tests for Thrivemind policy system — dataclasses, parsing, ordering, stages."""

import pytest

from library.tools.thrivemind import (
    DEFAULT_PROMPT,
    StagePolicy,
    PreprocessPolicy,
    PostprocessPolicy,
    OnMessagePolicy,
    ThinkingPolicy,
    ThrivemindPolicies,
    ThrivemindConfig,
    apply_preprocess,
    apply_postprocess,
    describe_processes,
    load_policies,
    order_colony,
    run_thinking_stage,
    write_process_description,
    _resolve_visibility,
)
from library.tools.identity import Identity
from unittest.mock import MagicMock
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colony(n: int = 4) -> list[Identity]:
    """Create a small test colony with varying cohesion/approval."""
    return [
        Identity(name=f"Ind_{i}", cohesion=float(i) * 0.2, approval=i)
        for i in range(n)
    ]


def _make_mock_ctx(config_dict: dict | None = None) -> MagicMock:
    """Create a mock InstanceContext for policy loading."""
    ctx = MagicMock()
    ctx.config = MagicMock(side_effect=lambda key: (config_dict or {}).get(key))
    ctx.read = MagicMock(return_value=None)
    ctx.write = MagicMock()
    ctx.exists = MagicMock(return_value=False)
    ctx.list = MagicMock(return_value=[])
    ctx.instance_id = "test-thrivemind"
    ctx.species_id = "thrivemind"
    ctx.compact_file = MagicMock(return_value=None)
    # LLM mock — return simple string
    llm_resp = MagicMock()
    llm_resp.message = "A principle of unity."
    llm_resp.tool_calls = []
    ctx.llm = MagicMock(return_value=llm_resp)
    return ctx


# ---------------------------------------------------------------------------
# StagePolicy validation
# ---------------------------------------------------------------------------


class TestStagePolicy:
    def test_valid_defaults(self):
        s = StagePolicy(name="test")
        assert s.stage_type == "individual"
        assert s.ordering == "random"
        assert s.visibility_after == "revealed"
        assert s.visibility_in_phase == "none"
        assert s.prompt_template == DEFAULT_PROMPT

    def test_valid_collective(self):
        s = StagePolicy(
            name="collective",
            stage_type="collective",
            ordering="cohesion_desc",
            visibility_after="private",
            visibility_in_phase="incremental",
        )
        assert s.stage_type == "collective"
        assert s.ordering == "cohesion_desc"

    def test_invalid_stage_type_raises(self):
        with pytest.raises(ValueError, match="Invalid stage_type"):
            StagePolicy(name="bad", stage_type="unknown")

    def test_invalid_ordering_raises(self):
        with pytest.raises(ValueError, match="Invalid ordering"):
            StagePolicy(name="bad", ordering="alphabetical")

    def test_invalid_visibility_after_raises(self):
        with pytest.raises(ValueError, match="Invalid visibility_after"):
            StagePolicy(name="bad", visibility_after="secret")

    def test_invalid_visibility_in_phase_raises(self):
        with pytest.raises(ValueError, match="Invalid visibility_in_phase"):
            StagePolicy(name="bad", visibility_in_phase="maybe")


# ---------------------------------------------------------------------------
# ThinkingPolicy validation
# ---------------------------------------------------------------------------


class TestThinkingPolicy:
    def test_max_3_stages(self):
        stages = [StagePolicy(name=f"s{i}") for i in range(4)]
        with pytest.raises(ValueError, match="At most 3"):
            ThinkingPolicy(stages=stages)

    def test_max_2_post_spawn(self):
        stages = [StagePolicy(name=f"ps{i}") for i in range(3)]
        with pytest.raises(ValueError, match="At most 2"):
            ThinkingPolicy(post_spawn=stages)

    def test_empty_is_valid(self):
        t = ThinkingPolicy()
        assert t.stages == []
        assert t.post_spawn == []


# ---------------------------------------------------------------------------
# ThrivemindPolicies.from_dict
# ---------------------------------------------------------------------------


class TestPoliciesFromDict:
    def test_empty_dict_returns_defaults(self):
        p = ThrivemindPolicies.from_dict({})
        assert p.on_message.preprocess.enabled is True
        assert p.on_message.preprocess.prompt_template == DEFAULT_PROMPT
        assert p.thinking.stages == []

    def test_none_returns_defaults(self):
        p = ThrivemindPolicies.from_dict(None)
        assert p.on_message.postprocess.enabled is True

    def test_custom_preprocess(self):
        p = ThrivemindPolicies.from_dict({
            "on_message": {
                "preprocess": {
                    "enabled": True,
                    "prompt_template": "Summarize: {message}",
                },
            },
        })
        assert p.on_message.preprocess.prompt_template == "Summarize: {message}"

    def test_disabled_postprocess(self):
        p = ThrivemindPolicies.from_dict({
            "on_message": {
                "postprocess": {"enabled": False},
            },
        })
        assert p.on_message.postprocess.enabled is False

    def test_thinking_stages_parsed(self):
        p = ThrivemindPolicies.from_dict({
            "thinking": {
                "stages": [
                    {"name": "reflect", "type": "individual", "ordering": "cohesion_asc"},
                    {"name": "dialogue", "type": "collective", "ordering": "approval_desc",
                     "visibility_in_phase": "incremental"},
                ],
            },
        })
        assert len(p.thinking.stages) == 2
        assert p.thinking.stages[0].name == "reflect"
        assert p.thinking.stages[0].ordering == "cohesion_asc"
        assert p.thinking.stages[1].stage_type == "collective"
        assert p.thinking.stages[1].visibility_in_phase == "incremental"

    def test_post_spawn_parsed(self):
        p = ThrivemindPolicies.from_dict({
            "thinking": {
                "post_spawn": [
                    {"name": "orientation", "type": "collective"},
                ],
            },
        })
        assert len(p.thinking.post_spawn) == 1

    def test_invalid_stage_skipped_with_warning(self):
        p = ThrivemindPolicies.from_dict({
            "thinking": {
                "stages": [
                    {"name": "good", "type": "individual"},
                    {"name": "bad", "type": "invalid_type"},
                ],
            },
        })
        assert len(p.thinking.stages) == 1
        assert p.thinking.stages[0].name == "good"

    def test_stage_type_alias(self):
        """Both 'type' and 'stage_type' should work."""
        p = ThrivemindPolicies.from_dict({
            "thinking": {
                "stages": [
                    {"name": "s1", "stage_type": "writer"},
                ],
            },
        })
        assert p.thinking.stages[0].stage_type == "writer"


# ---------------------------------------------------------------------------
# load_policies
# ---------------------------------------------------------------------------


class TestLoadPolicies:
    def test_no_config_returns_defaults(self):
        ctx = _make_mock_ctx()
        p = load_policies(ctx)
        assert p.on_message.preprocess.enabled is True
        assert p.thinking.stages == []

    def test_with_policies_config(self):
        ctx = _make_mock_ctx({
            "thrivemind": {
                "policies": {
                    "on_message": {
                        "preprocess": {"prompt_template": "Custom: {message}"},
                    },
                },
            },
        })
        p = load_policies(ctx)
        assert p.on_message.preprocess.prompt_template == "Custom: {message}"

    def test_non_dict_config_returns_defaults(self):
        ctx = _make_mock_ctx({"thrivemind": "not a dict"})
        p = load_policies(ctx)
        assert isinstance(p, ThrivemindPolicies)


# ---------------------------------------------------------------------------
# order_colony
# ---------------------------------------------------------------------------


class TestOrderColony:
    def test_cohesion_asc(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "cohesion_asc")
        cohesions = [ind.cohesion for ind in ordered]
        assert cohesions == sorted(cohesions)

    def test_cohesion_desc(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "cohesion_desc")
        cohesions = [ind.cohesion for ind in ordered]
        assert cohesions == sorted(cohesions, reverse=True)

    def test_approval_asc(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "approval_asc")
        approvals = [ind.approval for ind in ordered]
        assert approvals == sorted(approvals)

    def test_approval_desc(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "approval_desc")
        approvals = [ind.approval for ind in ordered]
        assert approvals == sorted(approvals, reverse=True)

    def test_combined_desc(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "combined_desc")
        scores = [ind.cohesion * (ind.approval + 1) for ind in ordered]
        assert scores == sorted(scores, reverse=True)

    def test_random_preserves_members(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "random", rng=random.Random(42))
        assert set(ind.name for ind in ordered) == set(ind.name for ind in colony)

    def test_unknown_ordering_falls_back_to_random(self):
        colony = _make_colony(4)
        ordered = order_colony(colony, "bogus", rng=random.Random(42))
        assert len(ordered) == 4


# ---------------------------------------------------------------------------
# _resolve_visibility
# ---------------------------------------------------------------------------


class TestResolveVisibility:
    def test_none_returns_empty(self):
        assert _resolve_visibility({"a": "text"}, "a", "none", []) == ""

    def test_private_returns_empty(self):
        assert _resolve_visibility({"a": "text"}, "a", "private", []) == ""

    def test_revealed_shows_all(self):
        outputs = {"Alice": "idea1", "Bob": "idea2", "Carol": "idea3"}
        result = _resolve_visibility(outputs, "Alice", "revealed", [])
        assert "Bob" in result
        assert "Carol" in result
        assert "Alice" in result  # revealed = everyone's output visible

    def test_incremental_shows_only_preceding(self):
        colony = [Identity(name="Alice"), Identity(name="Bob"), Identity(name="Carol")]
        outputs = {"Alice": "idea1", "Bob": "idea2", "Carol": "idea3"}
        # Carol should see Alice and Bob
        result = _resolve_visibility(outputs, "Carol", "incremental", colony)
        assert "Alice" in result
        assert "Bob" in result
        # Alice should see nobody
        result_alice = _resolve_visibility(outputs, "Alice", "incremental", colony)
        assert result_alice == ""

    def test_synthesis_key_excluded_from_revealed(self):
        outputs = {"Alice": "idea1", "_synthesis": "combined text"}
        result = _resolve_visibility(outputs, "Bob", "revealed", [])
        assert "_synthesis" not in result
        assert "Alice" in result


# ---------------------------------------------------------------------------
# describe_processes
# ---------------------------------------------------------------------------


class TestDescribeProcesses:
    def test_default_policies(self):
        cfg = ThrivemindConfig()
        policies = ThrivemindPolicies()
        desc = describe_processes(cfg, policies)
        assert "Colony Processes" in desc
        assert "on_message Flow" in desc
        assert "heartbeat Flow" in desc
        assert "PREPROCESS (default)" in desc
        assert "CONTRIBUTE" in desc  # default = no stages

    def test_custom_stages_described(self):
        cfg = ThrivemindConfig()
        policies = ThrivemindPolicies(
            thinking=ThinkingPolicy(
                stages=[
                    StagePolicy(name="reflect", stage_type="individual", ordering="cohesion_desc"),
                    StagePolicy(name="dialogue", stage_type="collective"),
                ],
            ),
        )
        desc = describe_processes(cfg, policies)
        assert "2 stage(s)" in desc
        assert "reflect" in desc
        assert "dialogue" in desc

    def test_custom_preprocess_described(self):
        cfg = ThrivemindConfig()
        policies = ThrivemindPolicies(
            on_message=OnMessagePolicy(
                preprocess=PreprocessPolicy(prompt_template="Custom: {message}"),
            ),
        )
        desc = describe_processes(cfg, policies)
        assert "PREPROCESS (custom)" in desc

    def test_post_spawn_described(self):
        cfg = ThrivemindConfig()
        policies = ThrivemindPolicies(
            thinking=ThinkingPolicy(
                post_spawn=[StagePolicy(name="orientation", stage_type="collective")],
            ),
        )
        desc = describe_processes(cfg, policies)
        assert "POST-SPAWN" in desc
        assert "orientation" in desc


# ---------------------------------------------------------------------------
# write_process_description
# ---------------------------------------------------------------------------


class TestWriteProcessDescription:
    def test_writes_to_processes_md(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policies = ThrivemindPolicies()
        write_process_description(ctx, cfg, policies)
        ctx.write.assert_called_once()
        path, content = ctx.write.call_args[0]
        assert path == "processes.md"
        assert "Colony Processes" in content


# ---------------------------------------------------------------------------
# apply_preprocess / apply_postprocess
# ---------------------------------------------------------------------------


class TestApplyPreprocess:
    def test_disabled_returns_raw(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PreprocessPolicy(enabled=False)
        result = apply_preprocess(ctx, "hello world", policy, cfg)
        assert result == "hello world"

    def test_default_returns_raw(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PreprocessPolicy(enabled=True, prompt_template=DEFAULT_PROMPT)
        result = apply_preprocess(ctx, "hello world", policy, cfg)
        assert result == "hello world"

    def test_custom_calls_llm(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PreprocessPolicy(enabled=True, prompt_template="Reframe: {message}")
        result = apply_preprocess(ctx, "hello world", policy, cfg)
        assert ctx.llm.called
        assert result == "A principle of unity."  # from mock


class TestApplyPostprocess:
    def test_disabled_returns_raw(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PostprocessPolicy(enabled=False)
        result = apply_postprocess(ctx, "draft reply", policy, cfg)
        assert result == "draft reply"

    def test_default_returns_raw(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PostprocessPolicy(enabled=True, prompt_template=DEFAULT_PROMPT)
        result = apply_postprocess(ctx, "draft reply", policy, cfg)
        assert result == "draft reply"

    def test_custom_calls_llm(self):
        ctx = _make_mock_ctx()
        cfg = ThrivemindConfig()
        policy = PostprocessPolicy(enabled=True, prompt_template="Polish: {candidate}")
        result = apply_postprocess(ctx, "draft reply", policy, cfg)
        assert ctx.llm.called


# ---------------------------------------------------------------------------
# run_thinking_stage
# ---------------------------------------------------------------------------


class TestRunThinkingStage:
    def test_individual_stage_produces_per_member_outputs(self):
        ctx = _make_mock_ctx()
        colony = _make_colony(3)
        cfg = ThrivemindConfig()
        stage = StagePolicy(name="reflect", stage_type="individual")
        reflections = {ind.name: "some reflection" for ind in colony}
        outputs = run_thinking_stage(ctx, colony, cfg, stage, "constitution text", reflections)
        # Each individual should have an output
        for ind in colony:
            assert ind.name in outputs

    def test_collective_stage_builds_shared_doc(self):
        ctx = _make_mock_ctx()
        colony = _make_colony(3)
        cfg = ThrivemindConfig()
        stage = StagePolicy(
            name="dialogue",
            stage_type="collective",
            ordering="approval_asc",
            visibility_in_phase="incremental",
        )
        reflections = {ind.name: "reflection" for ind in colony}
        outputs = run_thinking_stage(ctx, colony, cfg, stage, "constitution text", reflections)
        assert len(outputs) == 3

    def test_writer_stage_produces_synthesis(self):
        ctx = _make_mock_ctx()
        colony = _make_colony(3)
        cfg = ThrivemindConfig()
        stage = StagePolicy(name="synthesis", stage_type="writer")
        reflections = {}
        prior = {"Alice": "idea1", "Bob": "idea2"}
        outputs = run_thinking_stage(
            ctx, colony, cfg, stage, "constitution text", reflections,
            prior_stage_outputs=prior,
        )
        assert "_synthesis" in outputs

    def test_custom_prompt_used(self):
        ctx = _make_mock_ctx()
        colony = _make_colony(2)
        cfg = ThrivemindConfig()
        stage = StagePolicy(
            name="custom",
            stage_type="individual",
            prompt_template="Custom prompt for {individual_name}: {constitution}",
        )
        reflections = {ind.name: "" for ind in colony}
        outputs = run_thinking_stage(ctx, colony, cfg, stage, "our values", reflections)
        # Should have called LLM with custom prompt
        assert ctx.llm.called
        assert len(outputs) == 2


# ---------------------------------------------------------------------------
# ThrivemindPolicies.describe
# ---------------------------------------------------------------------------


class TestPoliciesDescribe:
    def test_default_describe(self):
        p = ThrivemindPolicies()
        desc = p.describe()
        assert "Active Policies" in desc
        assert "preprocess: enabled" in desc
        assert "postprocess: enabled" in desc
        assert "default (single contribution + synthesis)" in desc

    def test_custom_stages_describe(self):
        p = ThrivemindPolicies(
            thinking=ThinkingPolicy(
                stages=[
                    StagePolicy(name="s1", stage_type="individual", ordering="cohesion_desc"),
                ],
            ),
        )
        desc = p.describe()
        assert "1 pre-voting stage(s)" in desc
        assert "s1" in desc
        assert "cohesion_desc" in desc

    def test_disabled_preprocess_describe(self):
        p = ThrivemindPolicies(
            on_message=OnMessagePolicy(
                preprocess=PreprocessPolicy(enabled=False),
            ),
        )
        desc = p.describe()
        assert "preprocess: disabled" in desc
