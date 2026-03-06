"""OS schedule file generator — launchd (macOS), systemd (Linux), crontab fallback."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Literal

SchedulerType = Literal["launchd", "systemd", "crontab", "auto"]


def detect_scheduler() -> Literal["launchd", "systemd", "crontab"]:
    """Detect the OS scheduler type."""
    if platform.system() == "Darwin":
        return "launchd"
    if platform.system() == "Linux":
        # Check if systemd is available
        if Path("/run/systemd/system").exists():
            return "systemd"
    return "crontab"


# ------------------------------------------------------------------
# launchd (macOS)
# ------------------------------------------------------------------

_LAUNCHD_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{program_args}
    </array>
    <key>StartInterval</key>
    <integer>{interval}</integer>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StandardOutPath</key>
    <string>{log_dir}/symbiosis-{command}.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/symbiosis-{command}.err</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def _launchd_plist(
    label: str,
    command: str,
    interval: int,
    working_dir: str,
    log_dir: str,
    python: str,
    base_dir: str,
    config: str,
) -> str:
    args = [python, "-m", "symbiosis", command, "--base-dir", base_dir, "--config", config]
    args_xml = "\n".join(f"        <string>{a}</string>" for a in args)
    return _LAUNCHD_PLIST.format(
        label=label,
        program_args=args_xml,
        interval=interval,
        working_dir=working_dir,
        log_dir=log_dir,
        command=command,
    )


def generate_launchd(
    base_dir: Path,
    working_dir: Path,
    check_interval: int,
    work_interval: int,
    log_dir: str = "logs",
    config: str = "config/harness.yaml",
) -> dict[str, str]:
    """Return {filename: content} for launchd plist files."""
    python = sys.executable
    label_prefix = "io.symbiosis"
    out = {}

    for command, interval in [("check", check_interval), ("work", work_interval)]:
        label = f"{label_prefix}.{command}"
        filename = f"~/Library/LaunchAgents/{label}.plist"
        out[filename] = _launchd_plist(
            label=label,
            command=command,
            interval=interval,
            working_dir=str(working_dir),
            log_dir=log_dir,
            python=python,
            base_dir=str(base_dir),
            config=config,
        )

    return out


# ------------------------------------------------------------------
# systemd (Linux)
# ------------------------------------------------------------------

_SYSTEMD_SERVICE = """\
[Unit]
Description=Symbiosis {command} service
After=network.target

[Service]
Type=oneshot
WorkingDirectory={working_dir}
ExecStart={python} -m symbiosis {command} --base-dir {base_dir} --config {config}
StandardOutput=append:{log_dir}/symbiosis-{command}.log
StandardError=append:{log_dir}/symbiosis-{command}.err
"""

_SYSTEMD_TIMER = """\
[Unit]
Description=Symbiosis {command} timer

[Timer]
OnBootSec={interval}s
OnUnitActiveSec={interval}s
Unit=symbiosis-{command}.service

[Install]
WantedBy=timers.target
"""


def generate_systemd(
    base_dir: Path,
    working_dir: Path,
    check_interval: int,
    work_interval: int,
    log_dir: str = "logs",
    config: str = "config/harness.yaml",
) -> dict[str, str]:
    """Return {filename: content} for systemd service and timer unit files."""
    python = sys.executable
    out = {}

    for command, interval in [("check", check_interval), ("work", work_interval)]:
        service = _SYSTEMD_SERVICE.format(
            command=command,
            working_dir=str(working_dir),
            python=python,
            base_dir=str(base_dir),
            config=config,
            log_dir=log_dir,
        )
        timer = _SYSTEMD_TIMER.format(command=command, interval=interval)

        out[f"~/.config/systemd/user/symbiosis-{command}.service"] = service
        out[f"~/.config/systemd/user/symbiosis-{command}.timer"] = timer

    return out


# ------------------------------------------------------------------
# crontab
# ------------------------------------------------------------------

def _interval_to_cron(seconds: int) -> str:
    """Convert a seconds interval to a crontab expression (minute granularity)."""
    minutes = max(1, seconds // 60)
    if minutes == 1:
        return "* * * * *"
    if minutes < 60:
        return f"*/{minutes} * * * *"
    hours = minutes // 60
    return f"0 */{hours} * * *"


def generate_crontab(
    base_dir: Path,
    working_dir: Path,
    check_interval: int,
    work_interval: int,
    log_dir: str = "logs",
    config: str = "config/harness.yaml",
) -> dict[str, str]:
    """Return crontab entries as a single block."""
    python = sys.executable
    lines = [
        f"# Symbiosis scheduler — generated by `symbiosis schedule`",
        f"# Working directory: {working_dir}",
        "",
    ]

    for command, interval in [("check", check_interval), ("work", work_interval)]:
        cron_expr = _interval_to_cron(interval)
        log_file = f"{working_dir}/{log_dir}/symbiosis-{command}.log"
        cmd = (
            f"cd {working_dir} && "
            f"{python} -m symbiosis {command} "
            f"--base-dir {base_dir} --config {config} "
            f">> {log_file} 2>&1"
        )
        lines.append(f"{cron_expr} {cmd}")

    lines.append("")
    return {"crontab": "\n".join(lines)}


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def generate_schedule_files(
    base_dir: Path,
    working_dir: Path | None = None,
    check_interval: int = 300,
    work_interval: int = 60,
    log_dir: str = "logs",
    config: str = "config/harness.yaml",
    scheduler_type: SchedulerType = "auto",
) -> dict[str, str]:
    """
    Generate OS schedule config files.

    Returns a dict of {path: content} for the user to write.
    """
    if working_dir is None:
        working_dir = Path.cwd()

    sched = scheduler_type if scheduler_type != "auto" else detect_scheduler()

    if sched == "launchd":
        return generate_launchd(base_dir, working_dir, check_interval, work_interval, log_dir, config)
    if sched == "systemd":
        return generate_systemd(base_dir, working_dir, check_interval, work_interval, log_dir, config)
    return generate_crontab(base_dir, working_dir, check_interval, work_interval, log_dir, config)


def print_install_instructions(
    files: dict[str, str],
    scheduler_type: Literal["launchd", "systemd", "crontab"],
) -> None:
    """Print generated files and install instructions to stdout."""
    print(f"\nGenerated {scheduler_type} schedule files:\n")

    for path, content in files.items():
        print(f"=== {path} ===")
        print(content)

    print("\nInstall instructions:")
    if scheduler_type == "launchd":
        for path in files:
            expanded = path.replace("~", str(Path.home()))
            print(f"  cp <file> {expanded}")
        labels = [p.split("/")[-1].replace(".plist", "") for p in files]
        for label in labels:
            print(f"  launchctl load ~/Library/LaunchAgents/{label}.plist")
    elif scheduler_type == "systemd":
        for path in files:
            expanded = path.replace("~", str(Path.home()))
            print(f"  cp <file> {expanded}")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable --now symbiosis-check.timer symbiosis-work.timer")
    else:
        print("  Run: crontab -e")
        print("  Paste the crontab block above into your crontab.")
