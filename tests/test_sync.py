"""Tests for library.sync — instance memory sync to data repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from library.sync import sync_instances, _init_data_repo


@pytest.fixture
def instances_dir(tmp_path):
    """Create a fake instances directory with .md files."""
    inst = tmp_path / "instances"

    # Instance: alpha
    alpha = inst / "alpha" / "memory"
    alpha.mkdir(parents=True)
    (alpha / "thinking.md").write_text("# Thoughts\nSome deep thinking.")
    (alpha / "dreams.md").write_text("Dream of the ocean.")

    # Instance: beta
    beta = inst / "beta" / "memory"
    beta.mkdir(parents=True)
    (beta / "notes.md").write_text("# Notes\nImportant stuff.")

    # Binary file that should not be synced (sync only copies .md)
    (alpha / "net.pt").write_bytes(b"\x00\x01\x02")

    return inst


@pytest.fixture
def data_repo(tmp_path):
    """Create and return a path for the data repo."""
    return tmp_path / "data-repo"


class TestInitDataRepo:
    def test_creates_repo_with_gitignore(self, data_repo):
        _init_data_repo(data_repo, "main")

        assert (data_repo / ".git").is_dir()
        assert (data_repo / ".gitignore").exists()
        gitignore = (data_repo / ".gitignore").read_text()
        assert "*.pt" in gitignore
        assert "*.json" in gitignore


class TestSyncInstances:
    def test_first_sync_creates_repo_and_copies_files(self, instances_dir, data_repo):
        changed = sync_instances(instances_dir, data_repo, push=False)

        assert changed is True
        assert (data_repo / "alpha" / "thinking.md").exists()
        assert (data_repo / "alpha" / "dreams.md").exists()
        assert (data_repo / "beta" / "notes.md").exists()
        # Binary files should NOT be copied
        assert not (data_repo / "alpha" / "net.pt").exists()

    def test_content_matches(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, push=False)

        assert (data_repo / "alpha" / "thinking.md").read_text() == "# Thoughts\nSome deep thinking."

    def test_no_changes_returns_false(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, push=False)
        changed = sync_instances(instances_dir, data_repo, push=False)

        assert changed is False

    def test_detects_modified_files(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, push=False)

        # Modify a file
        (instances_dir / "alpha" / "memory" / "thinking.md").write_text("# Updated thoughts")

        changed = sync_instances(instances_dir, data_repo, push=False)
        assert changed is True
        assert (data_repo / "alpha" / "thinking.md").read_text() == "# Updated thoughts"

    def test_detects_new_files(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, push=False)

        # Add a new file
        (instances_dir / "alpha" / "memory" / "new.md").write_text("New file")

        changed = sync_instances(instances_dir, data_repo, push=False)
        assert changed is True
        assert (data_repo / "alpha" / "new.md").exists()

    def test_creates_git_commits(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, push=False)

        result = subprocess.run(
            ["git", "-C", str(data_repo), "log", "--oneline"],
            capture_output=True, text=True,
        )
        assert "Update instance memory" in result.stdout

    def test_nested_directories(self, instances_dir, data_repo):
        # Add nested .md file
        nested = instances_dir / "alpha" / "memory" / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "nested.md").write_text("Deep file")

        sync_instances(instances_dir, data_repo, push=False)
        assert (data_repo / "alpha" / "sub" / "deep" / "nested.md").exists()

    def test_missing_instances_dir(self, tmp_path, data_repo):
        changed = sync_instances(tmp_path / "nonexistent", data_repo, push=False)
        assert changed is False


class TestSyncWithPrefix:
    def test_prefix_creates_subdirectory(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, prefix="symbiosis", push=False)

        assert (data_repo / "symbiosis" / "alpha" / "thinking.md").exists()
        assert (data_repo / "symbiosis" / "beta" / "notes.md").exists()
        # Should NOT exist at root level
        assert not (data_repo / "alpha").exists()

    def test_prefix_no_changes_returns_false(self, instances_dir, data_repo):
        sync_instances(instances_dir, data_repo, prefix="symbiosis", push=False)
        changed = sync_instances(instances_dir, data_repo, prefix="symbiosis", push=False)
        assert changed is False

    def test_prefix_with_existing_repo(self, instances_dir, tmp_path):
        """Sync into a subdir of an already-initialized repo."""
        repo = tmp_path / "existing-repo"
        _init_data_repo(repo, "main")

        changed = sync_instances(instances_dir, repo, prefix="project-a", push=False)
        assert changed is True
        assert (repo / "project-a" / "alpha" / "thinking.md").exists()
