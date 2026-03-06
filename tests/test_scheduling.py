"""Tests for OS schedule file generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from symbiosis.scheduling import (
    detect_scheduler,
    generate_crontab,
    generate_launchd,
    generate_systemd,
    generate_schedule_files,
    _interval_to_cron,
)


class TestIntervalToCron:
    def test_one_minute(self):
        assert _interval_to_cron(60) == "* * * * *"

    def test_five_minutes(self):
        assert _interval_to_cron(300) == "*/5 * * * *"

    def test_one_hour(self):
        assert _interval_to_cron(3600) == "0 */1 * * *"

    def test_sub_minute_rounds_up(self):
        assert _interval_to_cron(30) == "* * * * *"


class TestGenerateLaunchd:
    def test_generates_two_files(self, tmp_path):
        files = generate_launchd(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        assert len(files) == 2
        labels = list(files.keys())
        assert any("check" in k for k in labels)
        assert any("work" in k for k in labels)

    def test_plist_content(self, tmp_path):
        files = generate_launchd(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        check_content = next(v for k, v in files.items() if "check" in k)
        assert "300" in check_content
        assert "symbiosis" in check_content
        assert sys.executable in check_content

    def test_work_interval(self, tmp_path):
        files = generate_launchd(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=120,
        )
        work_content = next(v for k, v in files.items() if "work" in k)
        assert "120" in work_content


class TestGenerateSystemd:
    def test_generates_four_files(self, tmp_path):
        files = generate_systemd(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        assert len(files) == 4
        names = list(files.keys())
        assert any("check.service" in k for k in names)
        assert any("check.timer" in k for k in names)
        assert any("work.service" in k for k in names)
        assert any("work.timer" in k for k in names)

    def test_timer_interval(self, tmp_path):
        files = generate_systemd(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        check_timer = next(v for k, v in files.items() if "check.timer" in k)
        assert "300s" in check_timer


class TestGenerateCrontab:
    def test_generates_one_entry(self, tmp_path):
        files = generate_crontab(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        assert "crontab" in files
        content = files["crontab"]
        assert "check" in content
        assert "work" in content

    def test_cron_expression_in_content(self, tmp_path):
        files = generate_crontab(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        content = files["crontab"]
        assert "*/5" in content  # 300s = 5 minutes
        assert "* * * * *" in content  # 60s = 1 minute


class TestGenerateScheduleFiles:
    def test_auto_returns_dict(self, tmp_path):
        files = generate_schedule_files(
            base_dir=tmp_path,
            working_dir=tmp_path,
            check_interval=300,
            work_interval=60,
        )
        assert isinstance(files, dict)
        assert len(files) > 0

    def test_explicit_crontab(self, tmp_path):
        files = generate_schedule_files(
            base_dir=tmp_path,
            working_dir=tmp_path,
            scheduler_type="crontab",
        )
        assert "crontab" in files

    def test_explicit_launchd(self, tmp_path):
        files = generate_schedule_files(
            base_dir=tmp_path,
            working_dir=tmp_path,
            scheduler_type="launchd",
        )
        assert any("check" in k for k in files)

    def test_explicit_systemd(self, tmp_path):
        files = generate_schedule_files(
            base_dir=tmp_path,
            working_dir=tmp_path,
            scheduler_type="systemd",
        )
        assert any("service" in k for k in files)
