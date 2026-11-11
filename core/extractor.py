import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            system_prompt + "\n\nMeeting transcript:\n\n{transcript}",
        ),
    ])

    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found."

    chain = build_chain(
        "You are an expert meeting analyst. Analyze the meeting transcript "
        "and extract all clearly identifiable action items.\n\n"
        "For each action item, provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, otherwise write 'Not specified')\n\n"
        "Rules:\n"
        "- Do not invent tasks, owners, or deadlines.\n"
        "- If the owner is unclear, write 'Not specified'.\n"
        "- Avoid duplicate action items.\n"
        "- Format the result as a numbered list.\n"
        "- If no action items are found, say 'No action items found.'"
    )

    return chain.invoke({"transcript": transcript})


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found."

    chain = build_chain(
        "You are an expert meeting analyst. Analyze the meeting transcript "
        "and extract all important decisions that were explicitly agreed "
        "upon or finalized during the meeting.\n\n"
        "Rules:\n"
        "- Include only actual decisions, not general discussions or suggestions.\n"
        "- Do not invent or assume decisions.\n"
        "- Avoid duplicate decisions.\n"
        "- Format the result as a numbered list.\n"
        "- If no key decisions are found, say 'No key decisions found.'"
    )

    return chain.invoke({"transcript": transcript})


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found."

    chain = build_chain(
        "You are an expert meeting analyst. Analyze the meeting transcript "
        "and extract unresolved questions, pending clarifications, and "
        "topics that require further discussion or follow-up.\n\n"
        "Rules:\n"
        "- Include only unresolved questions or pending topics.\n"
        "- Do not include questions that were already answered.\n"
        "- Do not invent questions.\n"
        "- Avoid duplicates.\n"
        "- Format the result as a numbered list.\n"
        "- If no open questions are found, say 'No open questions found.'"
    )

    return chain.invoke({"transcript": transcript})
