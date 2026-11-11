
import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


# ── Configuration ─────────────────────────────────────────────────────────────

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_CHUNK_MINUTES = 10


# ── YouTube Audio Download ────────────────────────────────────────────────────

def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and extract it as WAV using FFmpeg."""

    output_path = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        filename = Path(ydl.prepare_filename(info))
        wav_path = filename.with_suffix(".wav")

    if not wav_path.exists():
        raise FileNotFoundError(f"Downloaded WAV file not found: {wav_path}")

    return str(wav_path)


# ── Audio Conversion ─────────────────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """Convert an audio/video file into Whisper-compatible WAV format."""

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = input_file.with_name(
        f"{input_file.stem}_converted.wav"
    )

    audio = AudioSegment.from_file(str(input_file))

    # Whisper-compatible audio configuration
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_frame_rate(SAMPLE_RATE)

    audio.export(str(output_path), format="wav")

    return str(output_path)


# ── Audio Chunking ────────────────────────────────────────────────────────────

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES
) -> list[str]:
    """Split WAV audio into smaller chunks for Whisper transcription."""

    if chunk_minutes <= 0:
        raise ValueError("Chunk duration must be greater than zero.")

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{Path(wav_path).with_suffix('')}_chunk_{index}.wav"

        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# ── Main Audio Processing Pipeline ───────────────────────────────────────────

def process_input(source: str) -> list[str]:
    """
    Process YouTube URLs or uploaded audio/video files.

    Flow:
        YouTube URL / Local File
                ↓
        Download / Convert to WAV
                ↓
        Mono + 16 kHz
                ↓
        Audio Chunking
                ↓
        Return Chunk Paths
    """

    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks
