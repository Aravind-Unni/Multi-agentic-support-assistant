from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Relative to this file's location: src/RAG_setup.py -> parent is src/,
# parent.parent is the project root -> data/trendly_policy.md.
# This resolves correctly on Windows, inside Docker (where there's no
# C:\ drive), and on any other machine/OS this repo gets cloned onto.
POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "trendly_policy.md"

def setup_retriever():
    """Initializes the vector store and returns a retriever."""
    
    # 1. Load the policy text
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy_text = f.read()

    # 2. Split by Markdown headers to preserve context
    headers_to_split_on = [
        ("##", "Section"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(policy_text)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    vectorstore = FAISS.from_documents(md_splits, embeddings)


    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Instantiate the retriever once when this module is imported
policy_retriever = setup_retriever()