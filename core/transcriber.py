import os
from pathlib import Path

import whisper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Whisper configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Cached Whisper model (loaded only once)
_model = None


def load_model():
    """
    Load the local Whisper model.
    The model is cached so it does not reload for every audio chunk.
    """
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def get_gemini_llm():
    """
    Initialize Gemini for Hindi/Hinglish-to-English translation.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
    )


def normalize_language(language: str = "english"):
    """
    Convert user-friendly language choices into Whisper language codes.

    english  -> en
    hindi    -> hi
    hinglish -> None (automatic language detection)
    auto     -> None
    """
    if language is None:
        return None

    language = language.lower().strip()

    language_map = {
        "english": "en",
        "en": "en",
        "hindi": "hi",
        "hi": "hi",
        "hinglish": None,
        "auto": None,
        "auto-detect": None,
    }

    if language not in language_map:
        raise ValueError(
            f"Unsupported language: {language}. "
            "Use english, hindi, hinglish, or auto."
        )

    return language_map[language]


def transcribe_chunk_whisper(
    chunk_path: str,
    language: str = "english",
) -> str:
    """
    Transcribe one audio chunk using local Whisper.

    English uses English transcription.
    Hindi uses Hindi transcription.
    Hinglish uses automatic language detection.
    """
    if not Path(chunk_path).is_file():
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    model = load_model()
    whisper_language = normalize_language(language)

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language=whisper_language,
        fp16=False,
        verbose=False,
    )

    return result.get("text", "").strip()


def translate_to_english(transcript: str) -> str:
    """
    Translate a Hindi/Hinglish transcript into English using Gemini.

    This is performed after all audio chunks have been transcribed,
    avoiding a separate translation request for every audio chunk.
    """
    if not transcript or not transcript.strip():
        return ""

    llm = get_gemini_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """You are an expert Hindi and Hinglish to English translator.

Translate the provided meeting transcript into clear, natural English.

Rules:
- Preserve the original meaning and context.
- Translate Hindi sentences into English.
- Convert Hinglish sentences into natural English.
- Preserve technical terms, product names, and proper nouns.
- Do not summarize or omit any information.
- Do not add explanations or new information.
- Preserve the speaker's intended meaning.
- Return only the translated transcript.

Meeting transcript:

{transcript}"""
        ),
    ])

    translation_chain = prompt | llm | StrOutputParser()

    return translation_chain.invoke({
        "transcript": transcript
    }).strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
) -> str:
    """
    Transcribe one chunk using local Whisper.
    """
    return transcribe_chunk_whisper(
        chunk_path,
        language=language,
    )


def transcribe_all(
    chunks: list,
    language: str = "english",
    translate_to_english_output: bool = None,
) -> str:
    """
    Transcribe all audio chunks and combine their transcripts.

    Language behavior:
    - english: Whisper transcribes English.
    - hindi: Whisper transcribes Hindi.
    - hinglish: Whisper auto-detects speech, then Gemini translates
      the combined transcript into English.
    - auto: Whisper automatically detects the spoken language.

    Set translate_to_english_output=True to translate any supported
    transcription into English using Gemini.
    """
    if not chunks:
        raise ValueError("No audio chunks provided for transcription.")

    language_choice = (
        language.lower().strip() if isinstance(language, str) else "auto"
    )

    should_translate = (
        language_choice == "hinglish"
        if translate_to_english_output is None
        else translate_to_english_output
    )

    engine_description = "Whisper (local)"

    if should_translate:
        engine_description += " + Gemini (translation)"

    print(f"Using {engine_description} for transcription.")

    transcript_parts = []

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(
            chunk,
            language=language,
        )

        if text:
            transcript_parts.append(text)

    full_transcript = " ".join(transcript_parts).strip()

    if not full_transcript:
        raise ValueError(
            "Whisper could not generate a transcript from the audio."
        )

    # Translate once after combining all Whisper transcriptions
    if should_translate:
        print("Translating transcript into English using Gemini...")
        full_transcript = translate_to_english(full_transcript)

    print("Transcription complete.")

    return full_transcript.strip()
