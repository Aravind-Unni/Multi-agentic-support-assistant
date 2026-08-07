import json
import uuid
from langchain.tools import tool

@tool
def initiate_return(order_id: str, sku: str, reason: str, email: str) -> str:
    """Initiate and process a return for an item. 
    Requires order_id, sku, customer email, and reason for return.
    Only call this AFTER confirming eligibility with 'check_return'.
    Returns a unique return confirmation ID and pickup details."""

    # Generate a lightweight tracking reference
    return_id = f"RET-{uuid.uuid4().hex[:6].upper()}"
    
    return json.dumps({
        "status": "RETURN_INITIATED",
        "return_id": return_id,
        "order_id": order_id,
        "sku": sku,
        "pickup_window": "2 business days",
        "instructions": "Pack the item in its original box with tags attached. Carrier will attempt pickup twice."
    }, indent=2)