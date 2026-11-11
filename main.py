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
from core.rag_engine import build_rag_chain, ask_question


# ── Main Meeting Pipeline ─────────────────────────────────────────────────────

def run_pipeline(source: str, language: str = "en") -> dict:
    """
    Complete AI Meeting Assistant pipeline.

    Args:
        source: YouTube URL or local audio/video file path.
        language: Whisper language code.
                  "en"   -> English
                  "hi"   -> Hindi
                  None   -> Automatic language detection

    Returns:
        Dictionary containing meeting analysis and RAG chain.
    """

    print("\n" + "=" * 60)
    print("🎙️ AI MEETING ASSISTANT")
    print("=" * 60)

    # Step 1: Audio Processing
    print("\n[1/7] Processing audio...")

    chunks = process_input(source)

    print(f"Audio chunks created: {len(chunks)}")

    # Step 2: Local Whisper Transcription
    print("\n[2/7] Transcribing audio using local Whisper...")

    transcript = transcribe_all(
        chunks,
        language=language
    )

    if not transcript or not transcript.strip():
        raise ValueError("Transcription returned an empty result.")

    print("\n📝 Raw Transcription (first 300 characters):")
    print(transcript[:300])

    # Step 3: Generate Meeting Title using Gemini
    print("\n[3/7] Generating video title using Gemini...")

    title = generate_title(transcript)

    # Step 4: Generate Meeting Summary using Gemini
    print("\n[4/7] Generating video summary...")

    summary = summarize(transcript)

    # Step 5: Extract Action Items, Decisions and Questions
    print("\n[5/7] Extracting meeting insights...")

    action_items = extract_action_items(transcript)

    key_decisions = extract_key_decisions(transcript)

    open_questions = extract_questions(transcript)

    # Step 6: Build RAG Pipeline
    print("\n[6/7] Building RAG pipeline...")

    rag_chain = build_rag_chain(transcript)

    print("\n[7/7] Meeting processing completed successfully!")

    # Step 7: Return Complete Results
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    try:
        source = input(
            "\nEnter YouTube URL or local file path: "
        ).strip()

        print("\nSelect transcription language:")
        print("1. English")
        print("2. Hindi")
        print("3. Auto-detect")

        choice = input("Enter choice (1/2/3): ").strip()

        language_map = {
            "1": "en",
            "2": "hi",
            "3": None,
        }

        language = language_map.get(choice, "en")

        # Run complete meeting pipeline
        result = run_pipeline(source, language)

        # Display Meeting Results
        print("\n" + "=" * 60)
        print(f"📌 MEETING TITLE: {result['title']}")
        print("=" * 60)

        print("\n📋 MEETING SUMMARY")
        print("-" * 60)
        print(result["summary"])

        print("\n✅ ACTION ITEMS")
        print("-" * 60)
        print(result["action_items"])

        print("\n🔑 KEY DECISIONS")
        print("-" * 60)
        print(result["key_decisions"])

        print("\n❓ OPEN QUESTIONS")
        print("-" * 60)
        print(result["open_questions"])

        print("\n" + "=" * 60)

        # ── Phase 2: Chat with Meeting using RAG ──────────────────────────────

        print("\n💬 Chat with your meeting")
        print("Type 'exit', 'quit', or 'q' to stop.\n")

        rag_chain = result["rag_chain"]

        while True:

            question = input("You: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(rag_chain, question)

            print(f"\n🤖 Assistant: {answer}\n")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")

    except Exception as e:
        print(f"\n❌ Error in AI Meeting Assistant: {e}")
        raise
