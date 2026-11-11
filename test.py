# ── Load Environment Variables ────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY before importing core modules


# ── Project Imports ──────────────────────────────────────────────────────────

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)


# ── Configuration ─────────────────────────────────────────────────────────────

SOURCE = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"

# Whisper language:
# "en"   → English transcription
# "hi"   → Hindi transcription
# None   → Automatic language detection (useful for Hinglish)

WHISPER_LANGUAGE = "en"


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():

    # Step 1: Audio Processing
    print("\n" + "=" * 60)
    print("🎧 AUDIO PROCESSING")
    print("=" * 60)

    chunks = process_input(SOURCE)

    # Step 2: Local Whisper Transcription
    print("\n" + "=" * 60)
    print("🎙️ WHISPER TRANSCRIPTION")
    print("=" * 60)

    transcript = transcribe_all(
        chunks,
        language=WHISPER_LANGUAGE
    )

    print("\n📝 TRANSCRIPT")
    print("-" * 60)
    print(
        transcript[:500] + "..."
        if len(transcript) > 500
        else transcript
    )

    # Step 3: Gemini Title Generation
    print("\n" + "=" * 60)
    print("🤖 GEMINI AI PROCESSING")
    print("=" * 60)

    title = generate_title(transcript)

    # Step 4: Gemini Meeting Summarization
    summary = summarize(transcript)

    print("\n📌 MEETING TITLE")
    print("-" * 60)
    print(title)

    print("\n📋 MEETING SUMMARY")
    print("-" * 60)
    print(summary)

    # Step 5: Extract Action Items using Gemini
    action_items = extract_action_items(transcript)

    print("\n" + "=" * 60)
    print("✅ ACTION ITEMS")
    print("=" * 60)
    print(action_items)

    # Step 6: Extract Key Decisions using Gemini
    decisions = extract_key_decisions(transcript)

    print("\n" + "=" * 60)
    print("🔑 KEY DECISIONS")
    print("=" * 60)
    print(decisions)

    # Step 7: Extract Open Questions using Gemini
    questions = extract_questions(transcript)

    print("\n" + "=" * 60)
    print("❓ OPEN QUESTIONS")
    print("=" * 60)
    print(questions)

    print("\n" + "=" * 60)
    print("🎉 MEETING PROCESSING COMPLETED")
    print("=" * 60)


# ── Application Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error processing meeting: {e}")
        raise
