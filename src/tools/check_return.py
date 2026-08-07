import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from src.Agent.state import AgentState

DATA_PATH = Path(r"C:\Multi_agent\data\orders.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    _raw = json.load(f)

ORDERS = {o["order_id"]: o for o in _raw["orders"]}
CUSTOMERS = {c["customer_id"]: c for c in _raw["customers"]}


@tool
def check_return(order_id: str, sku: str, state: Annotated[AgentState, InjectedState]) -> str:
    """Check order status, delivery date, and item properties for a return.
    Requires order_id and item SKU. Customer must already be verified via
    get_order_status before this can be used.
    Returns item data and days elapsed since delivery for policy evaluation."""

    verified_customer_id = state.get("verified_customer_id")
    if not verified_customer_id:
        return "Customer is not verified yet. Call get_order_status first to verify."

    order = ORDERS.get(order_id)
    if not order:
        return f"Order '{order_id}' not found."

    if order["customer_id"] != verified_customer_id:
        return "This order does not belong to the verified customer. Refuse to discuss it."

    if order.get("status") == "cancelled":
        return "Order is CANCELLED. Returns cannot be raised against cancelled orders."

    if not order.get("delivered_at"):
        return f"Order status is '{order.get('status')}'. Item has not been delivered yet."

    item = next((i for i in order.get("items", []) if i.get("sku") == sku), None)
    if not item:
        return f"SKU '{sku}' not found in order {order_id}."

    delivered_date = datetime.fromisoformat(order["delivered_at"].replace("Z", "+00:00"))
    current_date = datetime.now(timezone.utc)
    days_since_delivery = (current_date - delivered_date).days

    return json.dumps({
        "order_id": order_id,
        "sku": sku,
        "category": item.get("category"),
        "final_sale": item.get("final_sale", False),
        "days_since_delivery": days_since_delivery,
        "delivered_at": order["delivered_at"],
    }, indent=2)