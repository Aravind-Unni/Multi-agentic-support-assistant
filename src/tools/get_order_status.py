import json
from pathlib import Path
from typing import Annotated

from langchain.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "orders.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    _raw = json.load(f)

ORDERS = {o["order_id"]: o for o in _raw["orders"]}
CUSTOMERS = {c["customer_id"]: c for c in _raw["customers"]}


@tool
def get_order_status(order_id: str, email: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Get order status/details by order_id. Requires email to verify
    ownership — return details only if email matches the order's account.
    On no match, give a generic "couldn't verify" message; never reveal
    whether the order_id itself exists."""
    order = ORDERS.get(order_id)
    if not order:
        msg = f"No order found with ID '{order_id}'. Please double-check the order ID."
        return Command(update={"messages": [{"role": "tool", "content": msg, "tool_call_id": tool_call_id}]})

    customer = CUSTOMERS.get(order["customer_id"])
    if not customer or customer["email"].lower() != email.strip().lower():
        msg = (
            f"I couldn't verify that '{order_id}' belongs to this email address. "
            f"Please confirm the order ID and the email used at checkout."
        )
        return Command(update={"messages": [{"role": "tool", "content": msg, "tool_call_id": tool_call_id}]})

    # Verified — persist identity in state so later tools trust it, not the LLM.
    return Command(update={
        "verified_email": customer["email"],
        "verified_customer_id": customer["customer_id"],
        "messages": [{"role": "tool", "content": json.dumps(order, indent=2), "tool_call_id": tool_call_id}],
    })