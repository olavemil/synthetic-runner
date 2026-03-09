"""Tests for standalone matrix CLI."""

from __future__ import annotations

from library.matrix_cli import main


def _input_from(values: list[str]):
    items = iter(values)
    return lambda _prompt="": next(items)


class TestMatrixCLI:
    def test_login_uses_existing_homeserver_and_writes_token(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("MATRIX_HOMESERVER=https://matrix-client.matrix.org\nEXISTING=1\n")

        monkeypatch.setattr(
            "builtins.input",
            _input_from(
                [
                    "HECATE_MATRIX_TOKEN",
                    "@hecate:matrix.org",
                ]
            ),
        )
        monkeypatch.setattr("library.matrix_cli.getpass.getpass", lambda _prompt="": "pw")

        captured = {}

        def _fake_login(homeserver: str, user: str, password: str, device_name: str = "symbiosis") -> str:
            captured.update(
                homeserver=homeserver,
                user=user,
                password=password,
                device_name=device_name,
            )
            return "token-123"

        monkeypatch.setattr("library.matrix_cli.MatrixAdapter.login", _fake_login)

        rc = main(["login", "--base-dir", str(tmp_path)])

        assert rc == 0
        assert captured == {
            "homeserver": "https://matrix-client.matrix.org",
            "user": "@hecate:matrix.org",
            "password": "pw",
            "device_name": "symbiosis",
        }
        env_text = env_path.read_text()
        assert "MATRIX_HOMESERVER=https://matrix-client.matrix.org" in env_text
        assert "HECATE_MATRIX_TOKEN=token-123" in env_text
        assert "EXISTING=1" in env_text

    def test_login_prompts_for_homeserver_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "builtins.input",
            _input_from(
                [
                    "https://matrix-client.matrix.org",
                    "HECATE_MATRIX_TOKEN",
                    "hecate",
                ]
            ),
        )
        monkeypatch.setattr("library.matrix_cli.getpass.getpass", lambda _prompt="": "pw")

        captured = {}

        def _fake_login(homeserver: str, user: str, password: str, device_name: str = "symbiosis") -> str:
            captured["homeserver"] = homeserver
            return "token-xyz"

        monkeypatch.setattr("library.matrix_cli.MatrixAdapter.login", _fake_login)

        rc = main(["login", "--base-dir", str(tmp_path)])

        assert rc == 0
        assert captured["homeserver"] == "https://matrix-client.matrix.org"
        env_text = (tmp_path / ".env").read_text()
        assert "MATRIX_HOMESERVER=https://matrix-client.matrix.org" in env_text
        assert "HECATE_MATRIX_TOKEN=token-xyz" in env_text
