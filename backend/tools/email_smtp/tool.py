from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tools.base import ToolDefinition

_DESCRIPTION = "Sends an email via SMTP. Always requires human approval before sending."


class EmailSendInput(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str
    body: str = Field(description="Plain text email body.")


class EmailSendTool(BaseTool):
    name: str = "email_smtp_send"
    description: str = _DESCRIPTION
    args_schema: type[BaseModel] = EmailSendInput

    def _run(self, to: str, subject: str, body: str) -> str:
        host = os.environ.get("SMTP_HOST")
        if not host:
            return "email_smtp is not configured: missing SMTP_HOST (and related SMTP_* env vars)."
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD")
        sender = os.environ.get("SMTP_FROM", user or "agent@localhost")

        message = MIMEText(body)
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = to

        try:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.sendmail(sender, [to], message.as_string())
        except Exception as exc:  # noqa: BLE001 — surfaced back to the agent as a tool result
            return f"Failed to send email: {exc}"
        return f"Email sent to {to}."


def build_tool(config: dict[str, Any]) -> BaseTool:
    return EmailSendTool()


TOOL_DEFINITION = ToolDefinition(
    id="email_smtp.send",
    name="Send Email",
    description=_DESCRIPTION,
    category="communication",
    factory=build_tool,
    sensitive=True,
    requires_credentials=["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"],
)
