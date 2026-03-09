"""Sync instance memory .md files to a separate data repository.

The data repo is a plain git repository containing only the instance memory
markdown files.  After `symbiosis work` runs, `symbiosis sync` copies any
changed .md files into the data repo and pushes.

Configuration lives in harness.yaml:

    sync:
      repo: /path/to/data-repo        # path to the data repository
      prefix: symbiosis                # subdirectory within the data repo
      branch: main                     # default: main

If the repo path doesn't exist and a remote URL is configured, it will be
cloned automatically.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _init_data_repo(repo_path: Path, branch: str) -> None:
    """Initialize a fresh data repo if it doesn't exist."""
    repo_path.mkdir(parents=True, exist_ok=True)
    _git(repo_path, "init", "-b", branch)
    # Initial commit so branch exists
    gitignore = repo_path / ".gitignore"
    gitignore.write_text("*.pt\n*.json\n*.db\ninbox/\n")
    _git(repo_path, "add", ".gitignore")
    _git(repo_path, "commit", "-m", "Initial commit", check=False)
    logger.info("Initialized data repo at %s", repo_path)


def _write_index_files(out_dir: Path) -> int:
    """Generate index.md files for each instance directory.

    Returns number of files written (only counts changed files).
    """
    written = 0

    for instance_dir in sorted(out_dir.iterdir()):
        if not instance_dir.is_dir() or instance_dir.name.startswith("."):
            continue

        md_files = sorted(instance_dir.rglob("*.md"))
        md_files = [f for f in md_files if f.name != "index.md"]
        if not md_files:
            continue

        lines = [
            "---",
            f'title: "{instance_dir.name}"',
            "---",
            "",
            f"# {instance_dir.name}",
            "",
        ]
        for md_file in md_files:
            rel = md_file.relative_to(instance_dir)
            display = rel.stem.replace("_", " ").replace("-", " ").title()
            link = str(rel.with_suffix(""))
            lines.append(f"- [{display}]({link})")

        # Link to published HTML files (graphs, visualizations)
        published_dir = instance_dir / "_published"
        if published_dir.is_dir():
            html_files = sorted(published_dir.glob("*.html"))
            if html_files:
                lines.append("")
                lines.append("### Visualizations")
                lines.append("")
                for html_file in html_files:
                    rel = html_file.relative_to(instance_dir)
                    display = html_file.stem.replace("_", " ").replace("-", " ").title()
                    lines.append(f"- [{display}]({rel})")

        content = "\n".join(lines) + "\n"
        index_path = instance_dir / "index.md"
        if index_path.exists() and index_path.read_text() == content:
            continue
        index_path.write_text(content)
        written += 1

    return written


def _add_frontmatter(content: str, stem: str) -> str:
    """Add Jekyll frontmatter if not already present."""
    if content.startswith("---\n"):
        return content
    title = stem.replace("_", " ").replace("-", " ").title()
    return f"---\ntitle: \"{title}\"\n---\n\n{content}"


def sync_instances(
    instances_dir: Path,
    repo_path: Path,
    branch: str = "main",
    remote: str | None = None,
    prefix: str | None = None,
    push: bool = True,
) -> bool:
    """Copy .md files from instances/ to data repo, commit, and optionally push.

    Args:
        prefix: Subdirectory within the data repo (e.g. "symbiosis").
                Files are written to repo_path/prefix/instance_id/.

    Returns True if changes were committed, False if nothing changed.
    """
    if not instances_dir.exists():
        logger.warning("Instances directory not found: %s", instances_dir)
        return False

    # Clone or init data repo if needed
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        # repo_path might be a subdir of an existing repo
        # Walk up to find .git
        parent = repo_path.parent
        while parent != parent.parent:
            if (parent / ".git").exists():
                git_dir = parent / ".git"
                break
            parent = parent.parent

    if not git_dir.exists():
        if remote:
            logger.info("Cloning data repo from %s", remote)
            subprocess.run(
                ["git", "clone", "-b", branch, remote, str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            _init_data_repo(repo_path, branch)

    # Determine the git working directory (root of the repo)
    git_root = git_dir.parent if git_dir.name == ".git" else repo_path

    # Determine output directory
    out_dir = repo_path / prefix if prefix else repo_path

    # Copy .md and .json files
    copied = 0
    for instance_dir in sorted(instances_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        memory_dir = instance_dir / "memory"
        if not memory_dir.is_dir():
            continue

        instance_id = instance_dir.name
        # Collect both .md and .json files
        files_to_copy = list(memory_dir.rglob("*.md")) + list(memory_dir.rglob("*.json"))
        for source_file in sorted(files_to_copy):
            rel = source_file.relative_to(memory_dir)
            dest = out_dir / instance_id / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Only copy if content differs
            raw_content = source_file.read_text(errors="replace")
            # Only add frontmatter for markdown files
            if source_file.suffix == ".md":
                new_content = _add_frontmatter(raw_content, rel.stem)
            else:
                new_content = raw_content
            if dest.exists() and dest.read_text(errors="replace") == new_content:
                continue

            dest.write_text(new_content)
            copied += 1

    # Generate index files for navigation
    copied += _write_index_files(out_dir)

    if copied == 0:
        logger.info("No changes to sync")
        return False

    logger.info("Copied %d changed file(s)", copied)

    # Stage, commit, push (use git_root for all git operations)
    _git(git_root, "add", "-A")

    # Check if there are actual staged changes
    result = _git(git_root, "diff", "--cached", "--quiet", check=False)
    if result.returncode == 0:
        logger.info("No git changes after staging")
        return False

    _git(git_root, "commit", "-m", "Update instance memory")
    logger.info("Committed changes")

    if push:
        result = _git(git_root, "remote", check=False)
        if result.stdout.strip():
            push_result = _git(git_root, "push", check=False)
            if push_result.returncode == 0:
                logger.info("Pushed to remote")
            else:
                logger.warning("Push failed: %s", push_result.stderr.strip())
        else:
            logger.info("No remote configured, skipping push")

    return True
