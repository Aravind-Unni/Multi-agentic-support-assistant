from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Using your absolute path, though consider Path(__file__).resolve().parent.parent / "data" / "trendly_policy.md" for deployment
POLICY_PATH = Path(r"C:\Multi_agent\data\trendly_policy.md")

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