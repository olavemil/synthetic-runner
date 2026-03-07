"""Standalone Matrix utility CLI."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

from symbiosis.harness.adapters.matrix import MatrixAdapter

PromptInput = Callable[[str], str]
PromptOutput = Callable[[str], None]

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _ask_text(
    prompt: str,
    *,
    input_fn: PromptInput | None = None,
    output_fn: PromptOutput = print,
    default: str | None = None,
    allow_empty: bool = True,
) -> str:
    if input_fn is None:
        input_fn = input

    while True:
        suffix = f" [{default}]" if default else ""
        raw = input_fn(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_empty:
            return ""
        output_fn("A value is required.")


def _upsert_env_values(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []

    key_to_index: dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = _ENV_LINE_RE.match(line)
        if match:
            key_to_index[match.group(1)] = idx

    for key, value in updates.items():
        entry = f"{key}={value}"
        if key in key_to_index:
            lines[key_to_index[key]] = entry
        else:
            lines.append(entry)

    path.write_text("\n".join(lines).rstrip() + "\n")


def _run_login(
    args: argparse.Namespace,
    *,
    input_fn: PromptInput | None = None,
    password_fn: Callable[[str], str] | None = None,
    output_fn: PromptOutput = print,
) -> int:
    if input_fn is None:
        input_fn = input
    if password_fn is None:
        password_fn = getpass.getpass

    base_dir = Path(args.base_dir)
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = base_dir / env_path

    existing_env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}

    homeserver = (
        args.homeserver
        or existing_env.get("MATRIX_HOMESERVER")
        or os.environ.get("MATRIX_HOMESERVER", "")
    )
    if not homeserver:
        homeserver = _ask_text(
            "Matrix homeserver (e.g. https://matrix-client.matrix.org)",
            input_fn=input_fn,
            output_fn=output_fn,
            allow_empty=False,
        )

    token_env_var = _ask_text(
        "Token env var name",
        input_fn=input_fn,
        output_fn=output_fn,
        default=args.env_var or "MATRIX_TOKEN",
        allow_empty=False,
    )
    username = _ask_text(
        "Matrix username (e.g. @bot:matrix.org or bot)",
        input_fn=input_fn,
        output_fn=output_fn,
        default=args.username,
        allow_empty=False,
    )

    password = ""
    while not password:
        password = password_fn("Matrix password: ").strip()
        if not password:
            output_fn("A value is required.")

    try:
        token = MatrixAdapter.login(
            homeserver=homeserver,
            user=username,
            password=password,
            device_name=args.device_name,
        )
    except Exception as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1

    _upsert_env_values(
        env_path,
        {
            "MATRIX_HOMESERVER": homeserver,
            token_env_var: token,
        },
    )
    output_fn(f"Login successful. Saved token to {env_path} as {token_env_var}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matrix utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("login", help="Log in to Matrix and store access token in .env")
    login.add_argument(
        "--base-dir", "-d",
        default=".",
        help="Base directory for resolving .env",
    )
    login.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (relative to --base-dir if not absolute)",
    )
    login.add_argument(
        "--homeserver",
        default="",
        help="Matrix homeserver URL (defaults to MATRIX_HOMESERVER in .env or env)",
    )
    login.add_argument(
        "--env-var",
        default="",
        help="Default token env var name to suggest in prompt",
    )
    login.add_argument(
        "--username",
        default="",
        help="Default username to suggest in prompt",
    )
    login.add_argument(
        "--device-name",
        default="symbiosis",
        help="Matrix device display name for login",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "login":
        return _run_login(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
