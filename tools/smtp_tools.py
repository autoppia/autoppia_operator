#!/usr/bin/env python3
"""SMTP helper tools for basic email notifications.

This module is MCP/CLI-friendly and reads SMTP credentials from environment:
- SMTP_HOST
- SMTP_PORT
- SMTP_USERNAME
- SMTP_PASSWORD
- SMTP_FROM
- SMTP_USE_SSL
- SMTP_USE_TLS
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate
from typing import Any


def _split_recipients(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in value.replace(";", ",").split(","):
        addr = item.strip()
        if addr:
            out.append(addr)
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "off", "no", "n"}:
        return False
    return default


def _smtp_config() -> dict[str, Any]:
    return {
        "host": os.getenv("SMTP_HOST", "").strip(),
        "port": _env_int("SMTP_PORT", 587),
        "username": os.getenv("SMTP_USERNAME", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip(),
        "default_from": os.getenv("SMTP_FROM", "").strip(),
        "timeout": _env_int("SMTP_TIMEOUT", 20),
        "use_ssl": _env_bool("SMTP_USE_SSL", False),
        "use_tls": _env_bool("SMTP_USE_TLS", True),
    }


def _default_to() -> list[str]:
    return _split_recipients(os.getenv("SMTP_DEFAULT_TO", ""))


@dataclass
class SmtpError(Exception):
    message: str


def _sanitize_error(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return "SMTP failure"
    return message


def _send_email(
    to_addresses: list[str],
    subject: str,
    body: str,
    *,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    reply_to: str | None = None,
    is_html: bool = False,
    from_override: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not to_addresses:
        raise SmtpError("No recipients provided (use --to or SMTP_DEFAULT_TO).")
    if not subject.strip():
        raise SmtpError("subject is required.")
    if not body.strip():
        raise SmtpError("body is required.")

    config = _smtp_config()
    if not config["host"]:
        raise SmtpError("SMTP_HOST is required.")
    sender = (from_override or config["default_from"]).strip()
    if not sender:
        raise SmtpError("SMTP_FROM is required (or pass --from-address).")

    cc = cc or []
    bcc = bcc or []
    recipients = sorted(set(to_addresses + cc + bcc))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_addresses)
    if cc:
        msg["Cc"] = ", ".join(cc)
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Date"] = formatdate(localtime=True)
    if is_html:
        msg.set_content("This message uses HTML content; open in an HTML-capable email client.")
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "from": sender,
            "to": to_addresses,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "html": is_html,
            "recipients": recipients,
            "transport": "ssl" if config["use_ssl"] else "starttls",
            "host": config["host"],
            "port": config["port"],
        }

    try:
        if config["use_ssl"]:
            context = ssl.create_default_context()
            client = smtplib.SMTP_SSL(config["host"], config["port"], timeout=config["timeout"], context=context)
        else:
            client = smtplib.SMTP(config["host"], config["port"], timeout=config["timeout"])
        with client:
            if config["use_tls"] and not config["use_ssl"]:
                client.starttls(context=ssl.create_default_context())
            if config["username"] and config["password"]:
                client.login(config["username"], config["password"])
            response = client.send_message(msg, from_addr=sender, to_addrs=recipients)
            # response is per recipient errors only, empty dict is success
            if response:
                return {"ok": False, "error_map": {str(k): str(v) for k, v in response.items()}}
            return {
                "ok": True,
                "message": "sent",
                "from": sender,
                "to": to_addresses,
                "cc": cc,
                "bcc": bcc,
                "recipients": recipients,
                "host": config["host"],
                "port": config["port"],
                "transport": "ssl" if config["use_ssl"] else "starttls",
            }
    except Exception as exc:
        return {"ok": False, "error": _sanitize_error(exc)}


def _check_connection() -> dict[str, Any]:
    config = _smtp_config()
    if not config["host"]:
        return {"ok": False, "error": "SMTP_HOST is required."}

    try:
        if config["use_ssl"]:
            with smtplib.SMTP_SSL(config["host"], config["port"], timeout=config["timeout"]) as client:
                code = client.noop()[0]
            return {"ok": True, "smtp_host": config["host"], "smtp_port": config["port"], "status_code": int(code)}
        with smtplib.SMTP(config["host"], config["port"], timeout=config["timeout"]) as client:
            code, _ = client.noop()
            if config["use_tls"] and code in {220, 250}:
                client.starttls(context=ssl.create_default_context())
            return {"ok": True, "smtp_host": config["host"], "smtp_port": config["port"], "status_code": int(code)}
    except Exception as exc:
        return {"ok": False, "error": _sanitize_error(exc)}


def cmd_send(args: argparse.Namespace) -> int:
    payload = _send_email(
        to_addresses=_split_recipients(args.to_addresses),
        subject=args.subject,
        body=args.body,
        cc=_split_recipients(args.cc),
        bcc=_split_recipients(args.bcc),
        reply_to=(args.reply_to or "").strip() or None,
        is_html=bool(args.is_html),
        from_override=(args.from_address or "").strip() or None,
        dry_run=bool(args.dry_run),
    )
    print(payload)
    if not payload.get("ok", False):
        return 1
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    payload = _check_connection()
    print(payload)
    if not payload.get("ok", False):
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMTP tools for operator email notifications")
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_cmd = subparsers.add_parser("send", help="send an email notification")
    send_cmd.add_argument("--to", dest="to_addresses", default=",".join(_default_to()), help="comma-separated recipient list")
    send_cmd.add_argument("--cc", default="", help="optional cc recipients")
    send_cmd.add_argument("--bcc", default="", help="optional bcc recipients")
    send_cmd.add_argument("--reply-to", default="", help="optional reply-to address")
    send_cmd.add_argument("--subject", required=True)
    send_cmd.add_argument("--body", required=True)
    send_cmd.add_argument("--from-address", default=os.getenv("SMTP_FROM", ""), help="override SMTP_FROM")
    send_cmd.add_argument("--html", dest="is_html", action="store_true", help="send body as HTML")
    send_cmd.add_argument("--dry-run", action="store_true", help="validate and print payload only")
    send_cmd.set_defaults(func=cmd_send)

    check_cmd = subparsers.add_parser("check", help="validate SMTP connection")
    check_cmd.set_defaults(func=cmd_check)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except SmtpError as exc:
        print(f"[smtp-tools] {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
