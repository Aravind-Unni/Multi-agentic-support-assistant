import json
import uuid
from langchain.tools import tool

@tool
def escalate_to_human(reason: str, summary: str, order_id: str = "UNKNOWN", email: str = "UNKNOWN") -> str:
    """Escalate the conversation to a human support agent.
    Use this immediately when:
    - A parcel is marked as lost or has no tracking movement (Lost-parcel claims).
    - You need to collect bank account details for a cash-on-delivery refund.
    - The policy document is silent on the user's specific question.
    - The user explicitly requests to speak with a human.
    Requires a specific reason and a brief summary of the user's issue.
    Returns a support ticket ID to provide to the user."""

    # Generate a lightweight ticket reference
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    
    # In a production environment, this payload would be logged to a database 
    # or sent via an email API to the support team's helpdesk.
    ticket_payload = {
        "status": "ESCALATED_TO_HUMAN",
        "ticket_id": ticket_id,
        "order_id": order_id,
        "email": email,
        "reason": reason,
        "summary": summary
    }
    
    return json.dumps(ticket_payload, indent=2)