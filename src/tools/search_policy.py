from langchain.tools import tool
from src.RAG_setup import policy_retriever 
@tool
def search_policy(query: str) -> str:
    """Search Trendly policy for general rules on shipping, returns, refunds, exchanges, and damaged items. 
        Use to answer general policy questions only. 
        Do NOT use this to evaluate a specific order's eligibility—use 'check_return' instead. 
        Returns exact text sections."""
    
    # The retriever automatically embeds the 'query' string using 
    # HuggingFace and performs the similarity search against the FAISS index.
    docs = policy_retriever.invoke(query)
    
    if not docs:
        return "No relevant policy found in the documentation for this query."
    
    # Format the retrieved chunks so the LLM knows which sections it's reading
    formatted_results = []
    for doc in docs:
        section_name = doc.metadata.get("Section", "General")
        formatted_results.append(f"--- Section: {section_name} ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_results)