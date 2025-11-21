import json
import http.server
import threading
import sys
from pathlib import Path
import tempfile
from typing import Callable, Optional

import streamlit as st


st.set_page_config(page_title="YouTube Loader", page_icon="📥")

MEDIA_ROOT = Path(tempfile.gettempdir()) / "karaoke_ano_novo"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)


class StepProgress:
    """Single progress bar sliced into equal steps."""

    def __init__(self, bar: st.delta_generator.DeltaGenerator, steps: int = 3):
        self.bar = bar
        self.steps = steps

    def update(self, step_index: int, fraction: float, text: str = ""):
        fraction = min(max(fraction, 0.0), 1.0)
        span = 1.0 / max(self.steps, 1)
        base = span * step_index
        value = base + fraction * span
        self.bar.progress(value, text=text)


if "logs" not in st.session_state:
    st.session_state["logs"] = []


def add_log(message: str):
    logs = st.session_state.setdefault("logs", [])
    logs.append(message)
    # keep log bounded
    if len(logs) > 200:
        del logs[0]


def render_terminal():
    st.divider()
    st.markdown("**Terminal**")
    logs = st.session_state.get("logs", [])
    content = "\n".join(logs) if logs else "No events yet."
    st.text_area("Logs", content, height=180, disabled=True)


def start_media_server(root: Path) -> str:
    """Spin up a lightweight HTTP file server to stream media without giant websocket payloads."""
    saved = st.session_state.get("media_server")
    if saved and Path(saved.get("root", "")) == root:
        return saved["base"]

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, directory=None, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):  # noqa: A003
            # Silence base HTTP logs
            return

    class QuietServer(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):  # noqa: D401
            exc = sys.exc_info()[1]
            if isinstance(exc, BrokenPipeError):
                return  # ignore client disconnects
            return super().handle_error(request, client_address)

    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(root), **kwargs)  # noqa: E731
    server = QuietServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    st.session_state["media_server"] = {"server": server, "thread": thread, "root": str(root), "base": base_url}
    add_log(f"Started media server at {base_url}")
    return base_url


def ensure_media_url(video_path: Path) -> str:
    """Return an HTTP URL for the video, starting the local server if needed."""
    base = start_media_server(video_path.parent)
    return f"{base}/{video_path.name}"


def _progress_hook(stepper: StepProgress, step_index: int = 0):
    def hook(d):
        if d.get("status") == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = min(max(downloaded / total, 0), 1)
                stepper.update(step_index, pct, text=f"Step {step_index + 1}/3: Downloading... {pct*100:.1f}%")
        elif d.get("status") == "finished":
            stepper.update(step_index, 1.0, text=f"Step {step_index + 1}/3: Download complete")
    return hook


def download_video(url: str, progress_callback: Optional[Callable] = None) -> str:
    """Download the YouTube video to a temp folder and return the file path."""
    temp_root = MEDIA_ROOT
    try:
        import yt_dlp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install yt-dlp (e.g. pip install yt-dlp)") from exc

    ydl_opts = {
        "outtmpl": str(temp_root / "%(id)s.%(ext)s"),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [progress_callback] if progress_callback else [],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filepath = Path(ydl.prepare_filename(info))

    expected_ext = ydl_opts.get("merge_output_format")
    final_path = raw_filepath.with_suffix(f".{expected_ext}") if expected_ext else raw_filepath

    if not final_path.exists():
        matches = sorted(
            temp_root.glob(f"{info.get('id', raw_filepath.stem)}.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            final_path = matches[0]

    if not final_path.exists():
        raise FileNotFoundError("Unable to locate the downloaded file.")

    return str(final_path)


@st.cache_resource(show_spinner="Loading speech-to-text model...")
def load_asr_pipeline():
    """Load a multilingual Whisper model for transcription."""
    try:
        from transformers import pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: install transformers and torch (e.g. pip install transformers torch)."
        ) from exc

    # Small model gives better multilingual quality and still runs locally on CPU (slower than tiny).
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",  # set to "cuda" if you have a GPU available
        chunk_length_s=None,  # let Whisper manage chunking to avoid experimental warnings
        ignore_warning=True,
        generate_kwargs={
            "task": "transcribe",
            "language": None,  # auto-detect (pt/en/es, etc)
        },
    )


@st.cache_data(show_spinner="Transcribing audio to subtitles...")
def generate_subtitles(video_path: str) -> tuple[str, str]:
    """Generate SRT subtitles using Whisper and return (srt_text, srt_file_path)."""
    voice_boosted_path = enhance_voice_for_asr(video_path)
    asr = load_asr_pipeline()
    result = asr(voice_boosted_path, return_timestamps="word")

    # Hugging Face Whisper returns chunks in a "chunks" list with start/end
    chunks = result.get("chunks") or []
    if not chunks:
        text = result.get("text", "").strip()
        if text:
            chunks = [
                {"text": text, "timestamp": (0.0, 0.0)},
            ]

    def to_srt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        ts = chunk.get("timestamp") or (0.0, 0.0)
        start = float(ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else 0.0)
        end_val = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 else None
        # Whisper may leave end None; default to small offset
        end = float(end_val) if end_val is not None else start + 2.0
        if end <= start:
            end = start + 0.5
        lines.append(str(idx))
        lines.append(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}")
        text = (chunk.get("text") or "").strip()
        lines.append(text or "[silence]")
        lines.append("")

    srt_text = "\n".join(lines).strip() + "\n"

    # Persist to temp for download/reuse
    srt_path = Path(video_path).with_suffix(".srt")
    srt_path.write_text(srt_text, encoding="utf-8")

    return srt_text, str(srt_path)


def enhance_voice_for_asr(video_path: str) -> str:
    """Apply a simple band-pass filter to reduce instrumental bleed before ASR."""
    try:
        from pydub import AudioSegment, effects  # type: ignore
    except ModuleNotFoundError:
        # Fallback: use original audio
        return video_path

    original = Path(video_path)
    voice_path = original.with_suffix(".voice.wav")

    try:
        audio = AudioSegment.from_file(video_path)
        # Remove rumble then trim high frequencies; focus on ~300-4000 Hz speech band
        audio = effects.high_pass_filter(audio, 180)
        audio = effects.low_pass_filter(audio, 4200)
        audio.export(voice_path, format="wav")
        return str(voice_path)
    except Exception:
        # If processing fails, still fall back to original
        return video_path


def render_video_with_subtitles(video_path: Path, srt_text: str):
    """Render a video with subtitles and a karaoke-style highlight of the spoken line."""
    cues = parse_srt_cues(srt_text)
    cues_json = json.dumps(cues)

    video_uri = ensure_media_url(video_path)

    html = f"""
    <style>
      .karaoke-container {{
        width: 100%;
        max-width: 960px;
        margin: 0 auto;
      }}
      .karaoke-line {{
        margin-top: 12px;
        padding: 12px 16px;
        background: #111;
        color: #f4f4f4;
        border-radius: 6px;
        font-size: 1.1rem;
        font-weight: 600;
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        overflow: hidden;
      }}
      .karaoke-line.active {{
        box-shadow: 0 0 12px rgba(255, 212, 71, 0.35);
      }}
      .karaoke-text {{
        --progress: 0%;
        background-image: linear-gradient(
          90deg,
          #ffd447 0%,
          #ffd447 var(--progress),
          #f4f4f4 var(--progress),
          #f4f4f4 100%
        );
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-fill-color: transparent;
        transition: background-position 0.05s linear;
        white-space: pre-wrap;
      }}
    </style>
    <div class="karaoke-container">
      <video id="karaoke-video" controls width="100%" crossorigin="anonymous">
        <source src="{video_uri}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div id="karaoke-line" class="karaoke-line">
        <span id="karaoke-text" class="karaoke-text">Loading subtitles...</span>
      </div>
    </div>
    <script>
      const cues = {cues_json};
      const video = document.getElementById("karaoke-video");
      const line = document.getElementById("karaoke-line");
      const textSpan = document.getElementById("karaoke-text");
      let lastText = "";

      function updateLine() {{
        if (!video || !cues.length) return;
        const t = video.currentTime;
        const cue = cues.find(c => t >= c.start && t <= c.end);
        const text = cue ? cue.text : "";
        const duration = cue ? Math.max(cue.end - cue.start, 0.001) : 1;
        const progress = cue ? Math.min(Math.max((t - cue.start) / duration, 0), 1) : 0;

        if (text !== lastText) {{
          textSpan.textContent = text || "♪";
          line.classList.toggle("active", !!cue);
          lastText = text;
        }}
        // Avoid Python f-string interpolation; build string in JS.
        textSpan.style.setProperty("--progress", (progress * 100) + "%");
        requestAnimationFrame(updateLine);
      }}

      video?.addEventListener("loadedmetadata", () => {{
        textSpan.textContent = "Ready to play";
        requestAnimationFrame(updateLine);
      }});
    </script>
    """
    st.components.v1.html(html, height=560)


def parse_srt_cues(srt_text: str) -> list[dict]:
    """Parse SRT into a list of {start, end, text} for highlighting."""
    def ts_to_seconds(ts: str) -> float:
        hh, mm, rest = ts.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

    cues = []
    block = []
    for line in srt_text.splitlines():
        if line.strip() == "":
            if block:
                cues.extend(_parse_block(block, ts_to_seconds))
                block = []
        else:
            block.append(line)
    if block:
        cues.extend(_parse_block(block, ts_to_seconds))
    return cues


def _parse_block(lines: list[str], ts_to_seconds: Callable[[str], float]) -> list[dict]:
    cues = []
    if not lines:
        return cues
    if "-->" in lines[0]:
        timing_line = lines[0]
        text_lines = lines[1:]
    elif len(lines) >= 2 and "-->" in lines[1]:
        timing_line = lines[1]
        text_lines = lines[2:]
    else:
        return cues

    try:
        start_ts, end_ts = [p.strip() for p in timing_line.split("-->")]
        start = ts_to_seconds(start_ts.replace(" ", ""))
        end = ts_to_seconds(end_ts.replace(" ", ""))
    except Exception:
        return cues

    text = " ".join([t.strip() for t in text_lines if t.strip()])
    cues.append({"start": start, "end": end, "text": text})
    return cues


def _ensure_subtitles(video_path: str, stepper: Optional[StepProgress] = None):
    """Generate subtitles for the provided video path if not already cached."""
    if (
        st.session_state.get("srt_video_path") == video_path
        and st.session_state.get("srt_text")
        and Path(st.session_state.get("srt_path", "")).exists()
    ):
        if stepper:
            stepper.update(1, 1.0, text="Step 2/3: Subtitles already cached")
            stepper.update(2, 1.0, text="Step 3/3: Ready to play with subtitles")
        return

    if stepper:
        stepper.update(1, 0.0, text="Step 2/3: Preparing transcription...")
    try:
        if stepper:
            stepper.update(1, 0.3, text="Step 2/3: Filtering audio for vocals...")
        srt_text, srt_path = generate_subtitles(video_path)
        if stepper:
            stepper.update(1, 1.0, text="Step 2/3: Subtitles ready")
    except Exception as exc:
        st.error(f"Could not generate subtitles: {exc}")
    else:
        st.session_state["srt_text"] = srt_text
        st.session_state["srt_path"] = srt_path
        st.session_state["srt_video_path"] = video_path
        st.success(f"Subtitles saved to {srt_path}")
        add_log(f"Subtitles saved to {srt_path}")
        if stepper:
            stepper.update(2, 1.0, text="Step 3/3: Ready to play with subtitles")


st.title("YouTube video loader")
st.write(
    "Paste a YouTube URL, download it to a temporary folder, and play it directly here."
)

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Download and show"):
    if not url.strip():
        st.warning("Please provide a valid YouTube link.")
    else:
        progress_bar = st.progress(0, text="Step 1/3: Starting download...")
        stepper = StepProgress(progress_bar, steps=3)
        stepper.update(0, 0.0, text="Step 1/3: Starting download...")
        try:
            video_path = download_video(url.strip(), progress_callback=_progress_hook(stepper, 0))
        except Exception as exc:
            st.error(f"Could not download the video: {exc}")
        else:
            stepper.update(0, 1.0, text="Step 1/3: Download complete")
            st.session_state["video_path"] = video_path
            st.success(f"Downloaded to {video_path}")
            add_log(f"Downloaded to {video_path}")
            _ensure_subtitles(video_path, stepper)
        # leave progress bar visible to show completion state

if "video_path" in st.session_state:
    path = Path(st.session_state["video_path"])
    if path.exists():
        if st.session_state.get("srt_video_path") != str(path):
            progress_bar = st.progress(0, text="Step 1/3: Using existing download...")
            stepper = StepProgress(progress_bar, steps=3)
            stepper.update(0, 1.0, text="Step 1/3: Using existing download")
            _ensure_subtitles(str(path), stepper)
        else:
            stepper = None

        if "srt_text" in st.session_state:
            render_video_with_subtitles(path, st.session_state["srt_text"])
        else:
            st.video(str(path))
        st.caption(f"Playing from: {path}")
        if st.session_state.get("last_play_logged") != str(path):
            add_log(f"Playing from: {path}")
            st.session_state["last_play_logged"] = str(path)

        if st.session_state.get("srt_text") and st.session_state.get("srt_path"):
            st.download_button(
                label="Download subtitles (.srt)",
                data=Path(st.session_state["srt_path"]).read_bytes(),
                file_name=Path(st.session_state["srt_path"]).name,
                mime="application/x-subrip",
            )
    else:
        st.info("Downloaded file was not found. Try downloading again.")

render_terminal()
