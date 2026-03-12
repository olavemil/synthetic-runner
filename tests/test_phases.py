"""Tests for library.tools.phases — scope-based tool filtering."""

from __future__ import annotations

import pytest

from library.tools.phases import (
    ARCHIVE_WRITE,
    KNOWLEDGE_READ,
    KNOWLEDGE_WRITE,
    GRAPH_READ,
    GRAPH_WRITE,
    MAP_READ,
    MAP_WRITE,
    CREATIVE_READ,
    CREATIVE_WRITE,
    THINK_SCOPES,
    ORGANIZE_SCOPES,
    SUBCONSCIOUS_SCOPES,
    DREAM_SCOPES,
    CREATE_SCOPES,
    SLEEP_SCOPES,
    filter_for_scopes,
    get_tools_for_scopes,
)


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


class TestFilterForScopes:
    def test_tool_included_when_scope_matches(self):
        schemas = [_schema("graph_write_tool")]
        scope_map = {"graph_write_tool": frozenset([GRAPH_WRITE])}
        result = filter_for_scopes(schemas, scope_map, frozenset([GRAPH_WRITE]))
        assert result == schemas

    def test_tool_excluded_when_scope_absent(self):
        schemas = [_schema("creative_write_tool")]
        scope_map = {"creative_write_tool": frozenset([CREATIVE_WRITE])}
        result = filter_for_scopes(schemas, scope_map, frozenset([GRAPH_WRITE, KNOWLEDGE_READ]))
        assert result == []

    def test_unknown_tool_excluded(self):
        schemas = [_schema("unknown_tool")]
        scope_map: dict = {}
        result = filter_for_scopes(schemas, scope_map, frozenset([GRAPH_WRITE]))
        assert result == []

    def test_multiple_tools_filtered_by_scope(self):
        schemas = [
            _schema("read_tool"),
            _schema("write_tool"),
            _schema("archive_tool"),
        ]
        scope_map = {
            "read_tool":    frozenset([KNOWLEDGE_READ]),
            "write_tool":   frozenset([KNOWLEDGE_WRITE]),
            "archive_tool": frozenset([ARCHIVE_WRITE]),
        }
        read_only = filter_for_scopes(schemas, scope_map, frozenset([KNOWLEDGE_READ]))
        assert len(read_only) == 1
        assert read_only[0]["function"]["name"] == "read_tool"

        read_write = filter_for_scopes(schemas, scope_map, frozenset([KNOWLEDGE_READ, KNOWLEDGE_WRITE]))
        assert len(read_write) == 2

        all_scopes = filter_for_scopes(schemas, scope_map, frozenset([KNOWLEDGE_READ, KNOWLEDGE_WRITE, ARCHIVE_WRITE]))
        assert len(all_scopes) == 3


class TestOrganizeScopeMaps:
    def test_read_tools_in_knowledge_read_scope(self):
        from library.tools.organize import ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP
        read_tools = [
            "organize_list_categories",
            "organize_list_topics",
            "organize_read_topic",
            "organize_list_archives",
            "organize_read_archive",
        ]
        for name in read_tools:
            schemas = [s for s in ORGANIZE_TOOL_SCHEMAS if s["function"]["name"] == name]
            result = filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([KNOWLEDGE_READ]))
            assert result, f"{name} should be in knowledge-read scope"

    def test_read_tools_excluded_without_knowledge_read(self):
        from library.tools.organize import ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP
        schemas = [s for s in ORGANIZE_TOOL_SCHEMAS
                   if s["function"]["name"] == "organize_list_categories"]
        result = filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([KNOWLEDGE_WRITE]))
        assert not result

    def test_archive_thoughts_in_archive_write_scope(self):
        from library.tools.organize import ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP
        schemas = [s for s in ORGANIZE_TOOL_SCHEMAS
                   if s["function"]["name"] == "organize_archive_thoughts"]
        assert filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([ARCHIVE_WRITE]))
        assert not filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([KNOWLEDGE_WRITE]))

    def test_write_topic_in_knowledge_write_scope(self):
        from library.tools.organize import ORGANIZE_TOOL_SCHEMAS, ORGANIZE_SCOPE_MAP
        schemas = [s for s in ORGANIZE_TOOL_SCHEMAS
                   if s["function"]["name"] == "organize_write_topic"]
        assert filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([KNOWLEDGE_WRITE]))
        assert not filter_for_scopes(schemas, ORGANIZE_SCOPE_MAP, frozenset([KNOWLEDGE_READ]))


class TestGraphMapScopeMaps:
    def test_graph_add_in_graph_write_scope(self):
        from library.tools.graph import GRAPH_TOOL_SCHEMAS, GRAPH_SCOPE_MAP
        write_tools = ["graph_add_node", "graph_add_edge", "graph_remove_node",
                       "graph_remove_edge", "graph_snapshot"]
        for name in write_tools:
            schemas = [s for s in GRAPH_TOOL_SCHEMAS if s["function"]["name"] == name]
            assert filter_for_scopes(schemas, GRAPH_SCOPE_MAP, frozenset([GRAPH_WRITE])), \
                f"{name} should be in graph-write"
            assert not filter_for_scopes(schemas, GRAPH_SCOPE_MAP, frozenset([GRAPH_READ])), \
                f"{name} should not be in graph-read"

    def test_graph_query_in_graph_read_scope(self):
        from library.tools.graph import GRAPH_TOOL_SCHEMAS, GRAPH_SCOPE_MAP
        read_tools = ["graph_query", "graph_describe"]
        for name in read_tools:
            schemas = [s for s in GRAPH_TOOL_SCHEMAS if s["function"]["name"] == name]
            assert filter_for_scopes(schemas, GRAPH_SCOPE_MAP, frozenset([GRAPH_READ])), \
                f"{name} should be in graph-read"
            assert not filter_for_scopes(schemas, GRAPH_SCOPE_MAP, frozenset([GRAPH_WRITE])), \
                f"{name} should not be in graph-write"

    def test_map_write_tools(self):
        from library.tools.activation_map import MAP_TOOL_SCHEMAS, MAP_SCOPE_MAP
        write_tools = ["map_define", "map_set", "map_set_region", "map_clear", "map_snapshot"]
        for name in write_tools:
            schemas = [s for s in MAP_TOOL_SCHEMAS if s["function"]["name"] == name]
            assert filter_for_scopes(schemas, MAP_SCOPE_MAP, frozenset([MAP_WRITE])), \
                f"{name} should be in map-write"
            assert not filter_for_scopes(schemas, MAP_SCOPE_MAP, frozenset([MAP_READ])), \
                f"{name} should not be in map-read"

    def test_map_read_tools(self):
        from library.tools.activation_map import MAP_TOOL_SCHEMAS, MAP_SCOPE_MAP
        read_tools = ["map_get", "map_describe"]
        for name in read_tools:
            schemas = [s for s in MAP_TOOL_SCHEMAS if s["function"]["name"] == name]
            assert filter_for_scopes(schemas, MAP_SCOPE_MAP, frozenset([MAP_READ])), \
                f"{name} should be in map-read"
            assert not filter_for_scopes(schemas, MAP_SCOPE_MAP, frozenset([MAP_WRITE])), \
                f"{name} should not be in map-write"


class TestNamedScopeSets:
    def test_think_scopes_has_archive_write(self):
        assert ARCHIVE_WRITE in THINK_SCOPES

    def test_think_scopes_has_no_creative(self):
        assert CREATIVE_READ not in THINK_SCOPES
        assert CREATIVE_WRITE not in THINK_SCOPES

    def test_organize_scopes_has_no_archive_write(self):
        assert ARCHIVE_WRITE not in ORGANIZE_SCOPES

    def test_organize_scopes_has_no_creative(self):
        assert CREATIVE_READ not in ORGANIZE_SCOPES
        assert CREATIVE_WRITE not in ORGANIZE_SCOPES

    def test_subconscious_scopes_is_read_only(self):
        write_scopes = {ARCHIVE_WRITE, KNOWLEDGE_WRITE, GRAPH_WRITE, MAP_WRITE,
                        CREATIVE_WRITE}
        assert not (SUBCONSCIOUS_SCOPES & write_scopes)

    def test_subconscious_scopes_includes_knowledge_and_graph_and_map(self):
        assert KNOWLEDGE_READ in SUBCONSCIOUS_SCOPES
        assert GRAPH_READ in SUBCONSCIOUS_SCOPES
        assert MAP_READ in SUBCONSCIOUS_SCOPES

    def test_dream_scopes_is_read_only(self):
        write_scopes = {ARCHIVE_WRITE, KNOWLEDGE_WRITE, GRAPH_WRITE, MAP_WRITE,
                        CREATIVE_WRITE}
        assert not (DREAM_SCOPES & write_scopes)

    def test_dream_scopes_excludes_structured_knowledge(self):
        # Dreams draw from graph/map patterns, not structured knowledge categories
        assert KNOWLEDGE_READ not in DREAM_SCOPES
        assert GRAPH_READ in DREAM_SCOPES
        assert MAP_READ in DREAM_SCOPES
        assert CREATIVE_READ in DREAM_SCOPES

    def test_create_scopes_has_creative_write(self):
        assert CREATIVE_WRITE in CREATE_SCOPES
        assert CREATIVE_READ in CREATE_SCOPES

    def test_create_scopes_has_no_write_access_except_creative(self):
        assert KNOWLEDGE_WRITE not in CREATE_SCOPES
        assert GRAPH_WRITE not in CREATE_SCOPES
        assert MAP_WRITE not in CREATE_SCOPES
        assert ARCHIVE_WRITE not in CREATE_SCOPES

    def test_create_scopes_has_read_access(self):
        assert KNOWLEDGE_READ in CREATE_SCOPES
        assert GRAPH_READ in CREATE_SCOPES
        assert MAP_READ in CREATE_SCOPES

    def test_sleep_scopes_has_archive_write_and_knowledge_write(self):
        assert ARCHIVE_WRITE in SLEEP_SCOPES
        assert KNOWLEDGE_READ in SLEEP_SCOPES
        assert KNOWLEDGE_WRITE in SLEEP_SCOPES

    def test_sleep_scopes_does_not_write_graph_or_map(self):
        assert GRAPH_WRITE not in SLEEP_SCOPES
        assert MAP_WRITE not in SLEEP_SCOPES

    def test_sleep_scopes_reads_graph_and_map_for_context(self):
        assert GRAPH_READ in SLEEP_SCOPES
        assert MAP_READ in SLEEP_SCOPES

    def test_sleep_scopes_has_no_creative(self):
        assert CREATIVE_READ not in SLEEP_SCOPES
        assert CREATIVE_WRITE not in SLEEP_SCOPES


class TestGetToolsForScopes:
    def test_think_scopes_includes_archive_thoughts(self):
        tools = get_tools_for_scopes(THINK_SCOPES)
        names = {t["function"]["name"] for t in tools}
        assert "organize_archive_thoughts" in names

    def test_organize_scopes_excludes_archive_thoughts(self):
        tools = get_tools_for_scopes(ORGANIZE_SCOPES)
        names = {t["function"]["name"] for t in tools}
        assert "organize_archive_thoughts" not in names
        assert "organize_write_topic" in names
        assert "organize_list_categories" in names

    def test_create_scopes_has_no_knowledge_write(self):
        tools = get_tools_for_scopes(CREATE_SCOPES)
        names = {t["function"]["name"] for t in tools}
        assert "organize_write_topic" not in names
        assert "organize_archive_thoughts" not in names
        assert "organize_list_categories" in names

    def test_graph_flag_adds_graph_tools_to_think(self):
        without = get_tools_for_scopes(THINK_SCOPES)
        with_graph = get_tools_for_scopes(THINK_SCOPES, graph=True)
        assert len(with_graph) > len(without)
        names = {t["function"]["name"] for t in with_graph}
        assert "graph_add_node" in names
        assert "graph_query" in names

    def test_create_scopes_with_graph_has_only_graph_read(self):
        tools = get_tools_for_scopes(CREATE_SCOPES, graph=True)
        names = {t["function"]["name"] for t in tools}
        assert "graph_query" in names
        assert "graph_describe" in names
        assert "graph_add_node" not in names
        assert "graph_add_edge" not in names

    def test_think_scopes_with_graph_has_graph_write(self):
        tools = get_tools_for_scopes(THINK_SCOPES, graph=True)
        names = {t["function"]["name"] for t in tools}
        assert "graph_add_node" in names

    def test_activation_map_flag_on_create_has_only_map_read(self):
        tools = get_tools_for_scopes(CREATE_SCOPES, activation_map=True)
        names = {t["function"]["name"] for t in tools}
        assert "map_get" in names
        assert "map_describe" in names
        assert "map_set" not in names
        assert "map_define" not in names

    def test_creative_flag_on_create_has_creative_write(self):
        tools = get_tools_for_scopes(CREATE_SCOPES, creative=True)
        names = {t["function"]["name"] for t in tools}
        assert "creative_new" in names
        assert "creative_list" in names

    def test_creative_flag_on_think_has_no_creative_write(self):
        tools = get_tools_for_scopes(THINK_SCOPES, creative=True)
        names = {t["function"]["name"] for t in tools}
        assert "creative_new" not in names
        assert "creative_edit" not in names
        assert "creative_list" not in names

    def test_creative_flag_on_organize_has_no_creative(self):
        tools = get_tools_for_scopes(ORGANIZE_SCOPES, creative=True)
        names = {t["function"]["name"] for t in tools}
        assert "creative_new" not in names
        assert "creative_list" not in names

    def test_think_with_all_flags(self):
        tools = get_tools_for_scopes(THINK_SCOPES, graph=True, activation_map=True)
        names = {t["function"]["name"] for t in tools}
        assert "organize_archive_thoughts" in names
        assert "graph_add_node" in names
        assert "map_set" in names
        assert "creative_new" not in names

    def test_organize_with_graph_excludes_archive(self):
        tools = get_tools_for_scopes(ORGANIZE_SCOPES, graph=True)
        names = {t["function"]["name"] for t in tools}
        assert "graph_add_node" in names
        assert "organize_write_topic" in names
        assert "organize_archive_thoughts" not in names
