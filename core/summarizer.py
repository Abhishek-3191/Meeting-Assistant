import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def split_transcript(transcript: str) -> list:
    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    llm = get_llm()

    # Step 1: Summarize individual transcript chunks (Map)
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """You are an expert meeting summarizer.

Summarize the provided portion of a meeting transcript.
Capture:
- Main topics discussed
- Important updates and progress
- Key points and concerns
- Decisions or conclusions mentioned

Rules:
- Preserve important names, dates, numbers, and technical details.
- Do not invent information.
- Keep the summary concise.
- Do not treat suggestions as finalized decisions.

Meeting transcript:
{text}"""
        ),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Step 2: Combine partial summaries into one final summary (Reduce)
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """You are an expert meeting summarizer.

Combine the provided partial summaries into one professional
and coherent meeting summary.

Instructions:
- Organize the summary into clear bullet points.
- Group related discussion points together.
- Preserve important names, dates, numbers, and technical details.
- Remove repetitive information.
- Do not introduce facts that are not present in the partial summaries.
- Clearly distinguish discussions, decisions, and pending matters.

Partial summaries:
{text}"""
        ),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """Based on the meeting transcript, generate a short,
professional meeting title.

Rules:
- Maximum 8 words.
- Reflect the main subject of the meeting.
- Do not use quotation marks.
- Only return the title, nothing else.

Meeting transcript:
{text}"""
        ),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:2000]}).strip()
