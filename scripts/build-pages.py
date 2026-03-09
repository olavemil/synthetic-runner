#!/usr/bin/env python3
"""Build a static site from instance memory .md files for GitHub Pages.

Copies all .md files from instances/*/memory/ into _site/ with a generated
index page and per-instance index pages.  Uses GitHub Pages' built-in Jekyll
for markdown rendering.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances"
SITE_DIR = ROOT / "_site"


def collect_md_files() -> dict[str, list[Path]]:
    """Return {instance_id: [relative .md paths]} from instances/*/memory/."""
    result: dict[str, list[Path]] = {}
    if not INSTANCES_DIR.exists():
        return result

    for instance_dir in sorted(INSTANCES_DIR.iterdir()):
        if not instance_dir.is_dir():
            continue
        memory_dir = instance_dir / "memory"
        if not memory_dir.is_dir():
            continue

        instance_id = instance_dir.name
        md_files = sorted(memory_dir.rglob("*.md"))
        if md_files:
            result[instance_id] = [f.relative_to(memory_dir) for f in md_files]

    return result


def build_site() -> None:
    """Generate the _site/ directory."""
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    # Jekyll config for GitHub Pages
    config = SITE_DIR / "_config.yml"
    config.write_text(
        "title: Symbiosis Instance Memory\n"
        "description: Live memory files from Symbiosis agent instances\n"
        "theme: jekyll-theme-minimal\n"
        "markdown: GFM\n"
    )

    instances = collect_md_files()

    # Root index
    lines = [
        "---",
        "title: Symbiosis Instance Memory",
        "---",
        "",
        "# Instance Memory",
        "",
    ]

    if not instances:
        lines.append("No instance memory files found.")
    else:
        for instance_id in instances:
            lines.append(f"- [{instance_id}]({instance_id}/)")

    (SITE_DIR / "index.md").write_text("\n".join(lines) + "\n")

    # Per-instance pages
    for instance_id, md_files in instances.items():
        instance_site = SITE_DIR / instance_id
        instance_site.mkdir(parents=True, exist_ok=True)

        # Instance index
        idx_lines = [
            "---",
            f"title: \"{instance_id}\"",
            "---",
            "",
            f"# {instance_id}",
            "",
            "[< Back to index](../)",
            "",
            "## Memory files",
            "",
        ]

        for md_path in md_files:
            # Display name: strip .md, replace / with " / "
            display = str(md_path.with_suffix("")).replace("/", " / ")
            # Link: use .html extension since Jekyll converts .md → .html
            link = str(md_path.with_suffix(""))
            idx_lines.append(f"- [{display}]({link})")

        (instance_site / "index.md").write_text("\n".join(idx_lines) + "\n")

        # Copy each .md file with Jekyll front matter
        memory_dir = INSTANCES_DIR / instance_id / "memory"
        for md_path in md_files:
            src = memory_dir / md_path
            dst = instance_site / md_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            content = src.read_text(errors="replace")
            title = md_path.stem.replace("_", " ").replace("-", " ").title()

            # Add front matter if not already present
            if not content.startswith("---"):
                content = (
                    f"---\n"
                    f"title: \"{title}\"\n"
                    f"---\n\n"
                    f"[< {instance_id}](../)\n\n"
                    f"{content}"
                )

            dst.write_text(content)

    # Count files
    total = sum(len(fs) for fs in instances.values())
    print(f"Built site: {len(instances)} instances, {total} files → {SITE_DIR}/")


if __name__ == "__main__":
    build_site()
