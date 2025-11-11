# edgeTtsCommandLine

A lightweight command-line wrapper around [edge-tts](https://github.com/rany2/edge-tts) that turns plain text or SubRip (`.srt`) subtitle files into high-quality spoken audio. The tool can optionally regenerate subtitle timing to match synthesized speech, making it convenient for creating narrated videos, accessibility assets, or localized voice-overs.

## Features

- 🔊 Convert plain text documents or `.srt` caption files into MP3 or WAV audio.
- 🗣️ Choose from the full catalog of Microsoft Edge neural voices built into `edge-tts`.
- ✂️ Automatically inserts configurable silence between synthesized segments.
- 🧾 Optionally regenerate `.srt` caption files when starting from plain text input.
- ♻️ Robust retry logic to handle transient synthesis errors gracefully.

## Requirements

- Python 3.9 or later.
- System installation of [FFmpeg](https://ffmpeg.org/) (required by `pydub` for audio processing).
- The Python packages listed in [`requirements.txt`](requirements.txt), including `edge-tts` and `pydub`.

## Installation

1. Clone the repository and move into the project directory:

   ```bash
   git clone https://github.com/<your-username>/edgeTtsCommandLine.git
   cd edgeTtsCommandLine
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Make sure FFmpeg is available on your `PATH`. On macOS you can install it with Homebrew (`brew install ffmpeg`); on Ubuntu, use `sudo apt install ffmpeg`.

## Usage

Run the converter by pointing it at an input file, selecting a voice, and providing the desired output path. The output format is inferred from the file extension (`.mp3` or `.wav`).

```bash
python tts_converter.py --input input.txt \
    --voice en-US-AriaNeural \
    --output output/audio.mp3 \
    --generate-srt \
    --silence 500 \
    --rate +0% --volume +0% --pitch +0Hz
```

### Supported inputs

- **Plain text (`.txt`)** – Each non-empty line becomes a separate utterance. The tool can optionally produce an `.srt` file aligned to the generated speech when `--generate-srt` is passed.
- **SubRip (`.srt`)** – Existing captions are synthesized in sequence. Silence between entries is enforced according to `--silence`, which acts as the minimum gap. Passing `--generate-srt` has no effect for this mode.

### Required options

- `--input` – Path to the source text or subtitle file.
- `--voice` – Name of the Edge neural voice to use. A comprehensive list is embedded in [`tts_converter.py`](tts_converter.py); you can also refer to the [`edge-tts` documentation](https://github.com/rany2/edge-tts#voices) for descriptions.
- `--output` – Destination file ending in `.mp3` or `.wav`.

### Helpful flags

| Flag | Description |
| ---- | ----------- |
| `--generate-srt` | Emit a new `.srt` file (only when starting from text input). |
| `--silence <ms>` | Amount of silence in milliseconds between segments (default: `750`). |
| `--rate <percent>` | Adjust speaking rate, e.g. `-10%` to slow down. |
| `--volume <percent>` | Adjust output volume, e.g. `+5%` to increase. |
| `--pitch <hz>` | Adjust voice pitch, e.g. `+2Hz`. |

### Examples

Convert a text script to MP3 and matching captions:

```bash
python tts_converter.py --input scripts/lesson.txt \
    --voice en-GB-LibbyNeural \
    --output build/lesson.mp3 \
    --generate-srt
```

Render an existing subtitle file into WAV audio while enforcing a 1 second minimum gap:

```bash
python tts_converter.py --input captions/show.srt \
    --voice en-US-GuyNeural \
    --output build/show.wav \
    --silence 1000
```

## Development

- Run `python tts_converter.py --help` to see the complete list of CLI options and defaults.
- Feel free to extend the script with new features or integrate it into larger automation pipelines.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue to discuss major changes, and submit a pull request when you're ready.

## License

This project is released under the [MIT License](LICENSE).

