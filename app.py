#!/usr/bin/env python3
"""
pdf2audiobook_gcloud.py

Usage:
    python pdf2audiobook_gcloud.py input.pdf output.mp3 [--voice en-US-Wavenet-D]

Dependencies:
    pip install PyPDF2 google-cloud-texttospeech pydub

You’ll also need:
    • A Google Cloud service account JSON key, with TEXT-TO-SPEECH API enabled
    • Set GOOGLE_APPLICATION_CREDENTIALS to its path
    • ffmpeg installed & on your PATH (for pydub to work)
"""

import io                              # For in-memory byte streams
import argparse                        # For parsing command-line arguments
from PyPDF2 import PdfReader          # PDF text extraction library
from pydub import AudioSegment        # Audio manipulation library (requires ffmpeg)
from google.cloud import texttospeech # Google Cloud TTS client library

# Maximum number of characters per TTS request (Google recommends ≤5000)
MAX_CHARS = 4500


def extract_text(pdf_path: str) -> str:
    """
    Read and extract all text from the given PDF file.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        str: Concatenated text from each page, separated by blank lines.
    """
    reader = PdfReader(pdf_path)  # Initialize PDF reader
    pages = []                    # Collect text from pages

    # Iterate through each page in the PDF
    for page in reader.pages:
        txt = page.extract_text()  # Extract raw text from the page
        if txt:
            pages.append(txt)      # Append non-empty text blocks

    # Join pages with double newline to preserve paragraph breaks
    return "\n\n".join(pages)


def chunk_text(text: str, max_chars: int = MAX_CHARS):
    """
    Split large text into smaller chunks suitable for TTS API limits.

    Splits on paragraph boundaries, but will slice overly long paragraphs as needed.

    Args:
        text (str): Full extracted text.
        max_chars (int): Maximum characters per chunk.

    Yields:
        str: Text chunks, each not exceeding max_chars.
    """
    paras = text.split("\n\n")  # Split into paragraphs
    chunk = ""

    for p in paras:
        # If adding this paragraph stays within the limit, append it
        if len(chunk) + len(p) + 2 <= max_chars:
            chunk += ("\n\n" + p) if chunk else p
        else:
            # Yield current chunk and start fresh
            yield chunk

            # If paragraph itself exceeds max_chars, slice it
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    yield p[i:i + max_chars]
                chunk = ""
            else:
                # Start a new chunk with this paragraph
                chunk = p

    # Yield any remaining text
    if chunk:
        yield chunk


def synthesize_chunk(client, text: str, voice_name: str, lang_code: str) -> bytes:
    """
    Convert a text chunk into speech audio bytes using Google Cloud TTS.

    Args:
        client: Initialized TextToSpeechClient.
        text (str): The text to synthesize.
        voice_name (str): The voice selection (e.g., 'en-US-Wavenet-D').
        lang_code (str): Language code (e.g., 'en-US').

    Returns:
        bytes: Binary audio content (MP3 format).
    """
    # Prepare the text input for synthesis
    input_ = texttospeech.SynthesisInput(text=text)

    # Configure the voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name
    )

    # Specify MP3 audio output
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the TTS request
    response = client.synthesize_speech(
        input=input_,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


def pdf_to_audiobook(pdf_path: str, output_mp3: str, voice_name: str, lang_code: str):
    """
    Full pipeline: Extract PDF text, split into chunks, synthesize each chunk,
    and concatenate into one MP3 audiobook.

    Args:
        pdf_path (str): Path to the source PDF.
        output_mp3 (str): Desired path for the output MP3.
        voice_name (str): Google TTS voice identifier.
        lang_code (str): Language code for synthesis.
    """
    print(f"⏳ Extracting text from {pdf_path} …")
    full_text = extract_text(pdf_path)  # Get raw text

    print("⏳ Initializing Google Cloud TTS client …")
    client = texttospeech.TextToSpeechClient()  # Auth via env var

    # Prepare an empty AudioSegment to concatenate into
    final_audio = AudioSegment.silent(duration=0)
    print("🔊 Converting text to speech in chunks …")

    # Loop through each text chunk and synthesize
    for i, chunk in enumerate(chunk_text(full_text), start=1):
        print(f" • Synthesizing chunk {i} ({len(chunk)} chars)…")
        audio_bytes = synthesize_chunk(client, chunk, voice_name, lang_code)
        # Load MP3 bytes into an AudioSegment
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        # Append to the final audio
        final_audio += segment

    print(f"💾 Exporting final audiobook to {output_mp3} …")
    # Export the combined AudioSegment as an MP3 file
    final_audio.export(output_mp3, format="mp3")
    print("✅ Done! Enjoy your audiobook 🎧")


if __name__ == "__main__":
    # Set up command-line interface
    parser = argparse.ArgumentParser(
        description="Convert PDF → MP3 audiobook via Google Cloud TTS"
    )
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument("mp3", help="Path to output MP3 file")
    parser.add_argument(
        "--voice",
        default="en-US-Wavenet-D",
        help="Google TTS voice name (e.g. en-US-Wavenet-D)"
    )
    parser.add_argument(
        "--lang",
        default="en-US",
        help="Language code (e.g. en-US)"
    )
    args = parser.parse_args()

    # Run the main conversion function with user-specified options
    pdf_to_audiobook(args.pdf, args.mp3, args.voice, args.lang)
