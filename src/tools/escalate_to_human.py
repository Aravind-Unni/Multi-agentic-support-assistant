import json
import uuid
import os
from langchain.tools import tool
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

@tool
def escalate_to_human(reason: str, summary: str, order_id: str = "UNKNOWN", email: str = "UNKNOWN") -> str:
    """Escalate to a human agent. Use when: parcel lost/no tracking
    movement, refund needs bank details, policy doesn't cover the
    question, or user explicitly asks for a human. Returns a ticket ID."""

    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    ticket_payload = {
        "status": "ESCALATED_TO_HUMAN",
        "ticket_id": ticket_id,
        "order_id": order_id,
        "email": email,
        "reason": reason,
        "summary": summary,
    }

    try:
        _send_ticket_email(ticket_payload)
    except Exception as e:
        # Never let email failure break the escalation itself
        ticket_payload["email_dispatch_error"] = str(e)

    return json.dumps(ticket_payload, indent=2)


def _send_ticket_email(payload: dict):
    message = Mail(
        from_email=os.environ["SUPPORT_FROM_EMAIL"],
        to_emails=os.environ["SUPPORT_INBOX_EMAIL"],
        subject=f"[{payload['ticket_id']}] Escalation — {payload['reason']}",
        plain_text_content=(
            f"Order: {payload['order_id']}\n"
            f"Customer: {payload['email']}\n"
            f"Reason: {payload['reason']}\n\n"
            f"Summary:\n{payload['summary']}"
        ),
    )
    sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    sg.send(message)