#!/usr/bin/env python3
"""Command line utility to synthesize speech with edge-tts.

This script can ingest either a plain text file (one utterance per line)
or a SubRip (``.srt``) subtitle file and produce an audio file in MP3 or WAV
format.  When a text file is used as input, the tool can optionally
produce an accompanying ``.srt`` file based on the generated audio.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import string
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Iterable, List, Sequence, Tuple

import edge_tts
from pydub import AudioSegment


@dataclass
class Caption:
    """Representation of a caption entry from an SRT file."""

    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create speech audio from text or SRT sources using edge-tts",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input text file or .srt subtitle file",
    )
    parser.add_argument(
        "--voice",
        required=True,
        help="Name of the edge-tts voice to use for synthesis",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output audio file (.mp3 or .wav)",
    )
    parser.add_argument(
        "--generate-srt",
        action="store_true",
        help="Generate a companion .srt file when the input is a text file",
    )
    parser.add_argument(
        "--silence",
        type=int,
        default=750,
        help=(
            "Silence duration in milliseconds. For text files this is inserted between "
            "lines; for SRT files it is the minimum silence enforced between captions."
        ),
    )
    parser.add_argument(
        "--rate",
        default=os.getenv("EDGE_TTS_RATE"),
        help="Optional speaking rate adjustment passed to edge-tts (e.g. '-10%')",
    )
    return parser.parse_args()


def ensure_output_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


PUNCTUATION_ONLY_CHARS = set(string.punctuation + " ，。？！一『…』“”“”：；《》（）")


def _is_soundless(text: str) -> bool:
    return all(char in PUNCTUATION_ONLY_CHARS for char in text)


def read_text_lines(path: str) -> List[str]:
    """Read non-empty, non-punctuation-only lines from a text file."""

    def _process(handle: Iterable[str]) -> List[str]:
        return [
            stripped
            for line in handle
            if (stripped := line.strip()) and not _is_soundless(stripped)
        ]

    encodings = ("utf-8", "utf-16", "utf-8-sig", "cp1252")
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as handle:
                return _process(handle)
        except UnicodeDecodeError:
            continue
    # Final attempt using default encoding (may still fail and propagate)
    with open(path, "r") as handle:
        return _process(handle)


def parse_timestamp(value: str) -> int:
    hours, minutes, seconds_millis = value.split(":")
    seconds, millis = seconds_millis.split(",")
    total_ms = (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )
    return total_ms


def format_timestamp(milliseconds: int) -> str:
    millis = milliseconds % 1000
    seconds_total = milliseconds // 1000
    seconds = seconds_total % 60
    minutes_total = seconds_total // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def parse_srt(path: str) -> List[Caption]:
    encodings = ("utf-8", "utf-16", "utf-8-sig", "cp1252")
    content = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as handle:
                content = handle.read()
                break
        except UnicodeDecodeError:
            continue
    if content is None:
        with open(path, "r") as handle:
            content = handle.read()

    captions: List[Caption] = []
    lines = [line.rstrip("\ufeff") for line in content.splitlines()]
    total_lines = len(lines)
    idx = 0
    while idx < total_lines:
        if not lines[idx].strip():
            idx += 1
            continue
        index_line = lines[idx].strip()
        idx += 1
        if idx >= total_lines:
            break
        timestamp_line = lines[idx].strip()
        idx += 1
        text_lines: List[str] = []
        while idx < total_lines and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        caption_text = "\n".join(text_lines).strip()
        try:
            start_raw, end_raw = [part.strip() for part in timestamp_line.split("-->")]
            start_ms = parse_timestamp(start_raw)
            end_ms = parse_timestamp(end_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp line: '{timestamp_line}'") from exc
        try:
            index_value = int(index_line)
        except ValueError:
            index_value = len(captions) + 1
        captions.append(Caption(index=index_value, start_ms=start_ms, end_ms=end_ms, text=caption_text))
        idx += 1  # Skip the blank line following the caption (if present)
    return captions


async def synthesize_segments(
    texts: Sequence[str],
    voice: str,
    temp_dir: str,
    rate: str | None = None,
    max_retries: int = 5,
) -> List[str]:
    file_paths: List[str] = []
    for i, text in enumerate(texts):
        temp_file = os.path.join(temp_dir, f"segment_{i}.mp3")
        attempt = 0
        while attempt < max_retries:
            try:
                communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
                await communicate.save(temp_file)
                file_paths.append(temp_file)
                break
            except Exception as exc:  # pragma: no cover - edge_tts errors are runtime dependent
                attempt += 1
                if attempt >= max_retries:
                    print(f"Failed to synthesize segment {i + 1}: {exc}")
                else:
                    print(f"Retrying segment {i + 1} due to error: {exc}")
    return file_paths


def build_audio_from_segments(
    audio_paths: Sequence[str],
    silence_duration: int,
    texts: Sequence[str],
    generate_srt: bool,
) -> Tuple[AudioSegment, List[Tuple[int, int, str]]]:
    combined = AudioSegment.silent(duration=0)
    timeline = 0
    srt_entries: List[Tuple[int, int, str]] = []
    for idx, (path, text) in enumerate(zip(audio_paths, texts)):
        segment_audio = AudioSegment.from_file(path)
        start = timeline
        end = start + len(segment_audio)
        combined += segment_audio
        if generate_srt:
            srt_entries.append((start, end, text))
        timeline = end
        if idx < len(audio_paths) - 1 and silence_duration > 0:
            combined += AudioSegment.silent(duration=silence_duration)
            timeline += silence_duration
    return combined, srt_entries


def build_audio_from_captions(
    captions: Sequence[Caption],
    audio_paths: Sequence[str],
    silence_duration: int,
) -> AudioSegment:
    combined = AudioSegment.silent(duration=0)
    timeline = 0
    for idx, (caption, path) in enumerate(zip(captions, audio_paths)):
        if idx == 0:
            gap = max(0, caption.start_ms)
        else:
            desired_gap = max(0, caption.start_ms - timeline)
            gap = max(desired_gap, silence_duration)
        if gap > 0:
            combined += AudioSegment.silent(duration=gap)
            timeline += gap
        segment_audio = AudioSegment.from_file(path)
        combined += segment_audio
        timeline += len(segment_audio)
    return combined


def write_srt(entries: Iterable[Tuple[int, int, str]], path: str) -> None:
    ensure_output_directory(path)
    with open(path, "w", encoding="utf-8") as handle:
        for index, (start, end, text) in enumerate(entries, start=1):
            handle.write(f"{index}\n")
            handle.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            handle.write(f"{text}\n\n")


def main() -> None:
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = os.path.abspath(args.output)
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext not in {".mp3", ".wav"}:
        raise ValueError("Output file must have an .mp3 or .wav extension")
    ensure_output_directory(output_path)

    silence_duration = max(0, int(args.silence))

    srt_entries: List[Tuple[int, int, str]] = []

    with TemporaryDirectory() as temp_dir:
        if input_path.lower().endswith(".srt"):
            captions = parse_srt(input_path)
            if not captions:
                raise ValueError("No captions found in the input SRT file")
            texts = [caption.text for caption in captions]
            audio_paths = asyncio.run(
                synthesize_segments(texts=texts, voice=args.voice, temp_dir=temp_dir, rate=args.rate)
            )
            if len(audio_paths) != len(texts):
                raise RuntimeError("Failed to synthesize all SRT segments")
            combined_audio = build_audio_from_captions(
                captions=captions, audio_paths=audio_paths, silence_duration=silence_duration
            )
            if args.generate_srt:
                print("Ignoring --generate-srt because the input is already an SRT file.")
        else:
            texts = read_text_lines(input_path)
            if not texts:
                raise ValueError("The input text file does not contain any lines to synthesize")
            audio_paths = asyncio.run(
                synthesize_segments(texts=texts, voice=args.voice, temp_dir=temp_dir, rate=args.rate)
            )
            if len(audio_paths) != len(texts):
                raise RuntimeError("Failed to synthesize all text lines")
            combined_audio, srt_entries = build_audio_from_segments(
                audio_paths=audio_paths,
                silence_duration=silence_duration,
                texts=texts,
                generate_srt=args.generate_srt,
            )

    combined_audio.export(output_path, format=output_ext.lstrip("."))
    print(f"Audio written to {output_path}")

    if srt_entries:
        srt_path = output_path + ".srt"
        write_srt(srt_entries, srt_path)
        print(f"SRT written to {srt_path}")


if __name__ == "__main__":
    main()
