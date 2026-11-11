import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    if not docs:
        return "No relevant transcript context was retrieved."

    return "\n\n".join(
        doc.page_content for doc in docs if doc.page_content
    )


def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "human",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting transcript
context provided below.

Rules:
- Do not use external knowledge.
- Do not invent or assume information.
- If the answer is not present in the context, say:
  "I could not find this information in the meeting transcript."
- Be concise, clear, and precise.
- If quoting someone, make it clear that it is a quote.
- If the context contains conflicting information, mention the conflict.

Context from meeting transcript:
{context}

User's question:
{question}"""
        ),
    ])


def build_rag_chain(transcript: str):
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    # Step 1: Store transcript chunks in ChromaDB
    vector_store = build_vector_store(transcript)

    # Step 2: Retrieve the 4 most relevant chunks
    retriever = get_retriever(vector_store, k=4)

    # Step 3: Initialize Gemini
    llm = get_llm()

    # Step 4: Define the RAG prompt
    prompt = get_rag_prompt()

    # Step 5: Build the LCEL RAG pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    # Load the previously persisted ChromaDB vector store
    vector_store = load_vector_store()

    # Retrieve relevant transcript chunks
    retriever = get_retriever(vector_store)

    llm = get_llm()

    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer
