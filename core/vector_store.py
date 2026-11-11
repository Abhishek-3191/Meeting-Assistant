
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ChromaDB local persistence configuration
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"

# Local HuggingFace embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Load the HuggingFace embedding model locally.

    The model converts transcript chunks into numerical vectors
    for semantic search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(transcript: str) -> Chroma:
    """
    Split the meeting transcript into chunks, generate embeddings,
    and store them in a persistent local ChromaDB collection.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    print("Building vector store...")

    # Step 1: Split transcript into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(transcript)

    # Step 2: Convert chunks into LangChain Documents
    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    # Step 3: Load local HuggingFace embeddings
    embeddings = get_embeddings()

    # Step 4: Create or replace the collection for this meeting
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Avoid retaining chunks from a previous meeting
    vector_store.reset_collection()

    # Step 5: Add current meeting documents
    vector_store.add_documents(documents=docs)

    print(f"Vector store created with {len(docs)} chunks.")

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load the previously persisted ChromaDB collection.
    """
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Return a retriever that fetches the top-k most relevant
    transcript chunks using similarity search.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
