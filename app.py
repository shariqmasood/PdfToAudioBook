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

import io
import argparse
from PyPDF2 import PdfReader
from pydub import AudioSegment
from google.cloud import texttospeech

# Google TTS recommends ≤5 000 characters per request
MAX_CHARS = 4500

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages.append(txt)
    return "\n\n".join(pages)

def chunk_text(text: str, max_chars: int = MAX_CHARS):
    paras = text.split("\n\n")
    chunk = ""
    for p in paras:
        if len(chunk) + len(p) + 2 <= max_chars:
            chunk += ("\n\n" + p) if chunk else p
        else:
            yield chunk
            if len(p) > max_chars:
                # break an over-long paragraph into slices
                for i in range(0, len(p), max_chars):
                    yield p[i:i+max_chars]
                chunk = ""
            else:
                chunk = p
    if chunk:
        yield chunk

def synthesize_chunk(client, text: str, voice_name: str, lang_code: str):
    input_ = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    resp = client.synthesize_speech(
        input=input_, voice=voice, audio_config=audio_config
    )
    return resp.audio_content

def pdf_to_audiobook(pdf_path: str, output_mp3: str, voice_name: str, lang_code: str):
    print(f"⏳ Extracting text from {pdf_path} …")
    full_text = extract_text(pdf_path)

    print("⏳ Initializing Google Cloud TTS client …")
    client = texttospeech.TextToSpeechClient()

    final = AudioSegment.silent(duration=0)
    print("🔊 Converting text to speech in chunks …")

    for i, chunk in enumerate(chunk_text(full_text), 1):
        print(f" • Synthesizing chunk {i} ({len(chunk)} chars)…")
        audio_bytes = synthesize_chunk(client, chunk, voice_name, lang_code)
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        final += segment

    print(f"💾 Exporting final audiobook to {output_mp3} …")
    final.export(output_mp3, format="mp3")
    print("✅ Done! Enjoy your audiobook 🎧")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF → MP3 audiobook via Google Cloud TTS")
    parser.add_argument("pdf", help="Path to input PDF")
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

    pdf_to_audiobook(args.pdf, args.mp3, args.voice, args.lang)
