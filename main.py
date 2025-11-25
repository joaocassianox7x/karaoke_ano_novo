import base64
import http.server
import threading
import sys
from pathlib import Path
import tempfile
from mimetypes import guess_type
import io
import math
import time
import numpy as np
import difflib
from typing import Callable, Optional

import streamlit as st


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


STREAMLIT_ACTIVE = _is_running_in_streamlit()

if STREAMLIT_ACTIVE:
    st.set_page_config(page_title="Carregador do YouTube", page_icon="📥")
    st.markdown(
        """
        <style>
          body { background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #0b1224 100%); color: #e5e7eb; }
          section.main > div { padding-top: 12px; }
          .block-container {
            padding: 28px 24px 32px 24px;
            border-radius: 14px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 10px 40px rgba(0,0,0,0.28);
          }
          h1, h2, h3, h4 { color: #f8fafc; }
          .stButton>button {
            background: linear-gradient(135deg, #22c55e, #10b981);
            color: #0f172a;
            border: none;
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(16,185,129,0.35);
          }
          .stButton>button:hover { filter: brightness(1.05); }
          .stProgress > div > div {
            background: linear-gradient(90deg, #3b82f6, #22c55e);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

DEFAULT_MEDIA_ROOT = Path(tempfile.gettempdir()) / "karaoke_ano_novo"
DEFAULT_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_ASR_MODEL = "openai/whisper-tiny"

COMPONENT_DIR = (Path(__file__).parent / "web_recorder").resolve()
if STREAMLIT_ACTIVE:
    karaoke_recorder_component = st.components.v1.declare_component(
        "karaoke_recorder",
        path=str(COMPONENT_DIR),
    )
else:
    karaoke_recorder_component = lambda *args, **kwargs: None  # type: ignore

FALLBACK_LOGS: list[str] = []


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


if STREAMLIT_ACTIVE:
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if "video_temp_dir" not in st.session_state:
        st.session_state["video_temp_dir"] = str(DEFAULT_MEDIA_ROOT)
    if "subtitle_temp_dir" not in st.session_state:
        st.session_state["subtitle_temp_dir"] = str(DEFAULT_MEDIA_ROOT)
    if "asr_model_name" not in st.session_state:
        st.session_state["asr_model_name"] = DEFAULT_ASR_MODEL


def add_log(message: str):
    if STREAMLIT_ACTIVE:
        logs = st.session_state.setdefault("logs", [])
        logs.append(message)
        if len(logs) > 200:
            del logs[0]
    else:
        FALLBACK_LOGS.append(message)
        if len(FALLBACK_LOGS) > 200:
            del FALLBACK_LOGS[0]


def _ensure_directory(path_str: str, fallback: Path) -> Path:
    """Create and return a resolved path; fallback to default on failure."""
    try:
        path = Path(path_str).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    except Exception as exc:
        add_log(f"Não foi possível usar {path_str}: {exc}. Voltando para {fallback}.")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


def get_video_temp_dir() -> Path:
    if not STREAMLIT_ACTIVE:
        return _ensure_directory(str(DEFAULT_MEDIA_ROOT), DEFAULT_MEDIA_ROOT)
    return _ensure_directory(st.session_state.get("video_temp_dir") or DEFAULT_MEDIA_ROOT, DEFAULT_MEDIA_ROOT)


def get_subtitle_temp_dir() -> Path:
    if not STREAMLIT_ACTIVE:
        return _ensure_directory(str(DEFAULT_MEDIA_ROOT), DEFAULT_MEDIA_ROOT)
    base = (
        st.session_state.get("subtitle_temp_dir")
        or st.session_state.get("video_temp_dir")
        or DEFAULT_MEDIA_ROOT
    )
    return _ensure_directory(base, DEFAULT_MEDIA_ROOT)


def get_asr_model_name() -> str:
    if not STREAMLIT_ACTIVE:
        return DEFAULT_ASR_MODEL
    return st.session_state.get("asr_model_name", DEFAULT_ASR_MODEL)


def render_terminal():
    st.divider()
    st.markdown("**Terminal**")
    logs = st.session_state.get("logs", [])
    content = "\n".join(logs) if logs else "Nenhum evento ainda."
    st.text_area("Registros", content, height=180, disabled=True)


def start_media_server(root: Path) -> str:
    """Spin up a lightweight HTTP file server to stream media without giant websocket payloads."""
    saved = st.session_state.get("media_server")
    if saved and Path(saved.get("root", "")) == root:
        return saved["base"]

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def __init__(self, *args, directory=None, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):  # noqa: A003
            # Silence base HTTP logs
            return

        def end_headers(self):
            # Allow media to be fetched from the Streamlit iframe (different port)
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    class QuietServer(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):  # noqa: D401
            exc = sys.exc_info()[1]
            # Ignore common disconnect errors when the browser stops the stream mid-transfer
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                return
            return super().handle_error(request, client_address)

    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(root), **kwargs)  # noqa: E731
    server = QuietServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    st.session_state["media_server"] = {"server": server, "thread": thread, "root": str(root), "base": base_url}
    add_log(f"Servidor de mídia iniciado em {base_url}")
    return base_url


def _streamlit_media_url(path: Path) -> Optional[str]:
    """
    Register the media file with Streamlit's own media endpoint so it is served
    from the same origin/port as the app (works on hosted/cloud environments).
    """
    mgr = None
    try:
        from streamlit.runtime.media_file_manager import media_file_manager as mgr  # type: ignore
    except Exception:
        try:
            from streamlit.media_file_manager import media_file_manager as mgr  # type: ignore
        except Exception:
            try:
                import streamlit.runtime as rt  # type: ignore
                mgr = rt.get_instance().media_file_manager  # type: ignore
            except Exception:
                add_log("Gerenciador de mídia do Streamlit indisponível; usando servidor local")
                return None

    mimetype, _ = guess_type(str(path))
    try:
        media_id = mgr.add(str(path), mimetype or "application/octet-stream", file_name=path.name)
        url = mgr.get_url(media_id)
        add_log(f"Servindo pelo endpoint de mídia do Streamlit: {url}")
        return url
    except Exception as exc:
        add_log(f"Não foi possível registrar mídia no Streamlit: {exc}")
        return None


def ensure_media_url(video_path: Path) -> str:
    """
    Return an HTTP URL for the video. Prefer Streamlit's built-in media handler
    (same origin), and fall back to the lightweight local HTTP server.
    """
    # Best effort: same-origin media avoids mixed-content / port exposure issues
    media_url = _streamlit_media_url(video_path)
    if media_url:
        return media_url

    base = start_media_server(video_path.parent)
    return f"{base}/{video_path.name}"


def _progress_hook(stepper: StepProgress, step_index: int = 0):
    def hook(d):
        if d.get("status") == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = min(max(downloaded / total, 0), 1)
                stepper.update(step_index, pct, text=f"Etapa {step_index + 1}/3: Baixando... {pct*100:.1f}%")
        elif d.get("status") == "finished":
            stepper.update(step_index, 1.0, text=f"Etapa {step_index + 1}/3: Download concluído")
    return hook


def download_video(url: str, progress_callback: Optional[Callable] = None) -> str:
    """Download the YouTube video to a temp folder and return the file path."""
    temp_root = get_video_temp_dir()
    try:
        import yt_dlp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Dependência ausente: instale yt-dlp (ex.: pip install yt-dlp).") from exc

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
        raise FileNotFoundError("Não foi possível localizar o arquivo baixado.")

    return str(final_path)


@st.cache_resource(show_spinner="Carregando modelo de transcrição de voz...")
def load_asr_pipeline(model_name: str):
    """Load a multilingual Whisper model for transcription."""
    try:
        from transformers import pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dependência ausente: instale transformers e torch (ex.: pip install transformers torch)."
        ) from exc

    # Small model gives better multilingual quality and still runs locally on CPU (slower than tiny).
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device="cpu",  # set to "cuda" if you have a GPU available
        chunk_length_s=None,  # let Whisper manage chunking to avoid experimental warnings
        ignore_warning=True,
        generate_kwargs={
            "task": "transcribe",
            "language": None,  # auto-detect (pt/en/es, etc)
        },
    )


@st.cache_data(show_spinner="Transcrevendo áudio para legendas...")
def generate_subtitles(video_path: str, model_name: str, subtitle_dir: str) -> tuple[str, str]:
    """Generate SRT subtitles using Whisper and return (srt_text, srt_file_path)."""
    voice_boosted_path = enhance_voice_for_asr(video_path)
    asr = load_asr_pipeline(model_name)
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
        lines.append(text or "[silêncio]")
        lines.append("")

    srt_text = "\n".join(lines).strip() + "\n"

    # Persist to temp for download/reuse
    target_dir = _ensure_directory(subtitle_dir, DEFAULT_MEDIA_ROOT)
    srt_path = target_dir / f"{Path(video_path).stem}.srt"
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
    """
    Render a video and feed subtitles via a native <track> element. The track
    is created from the SRT on the fly (converted to WebVTT) so browsers show
    the captions without custom JS syncing. Recording starts automatically when
    the video plays.
    """
    video_uri = ensure_media_url(video_path)
    cues = parse_srt_cues(srt_text)
    vtt_text = srt_to_webvtt(srt_text)

    result = karaoke_recorder_component(
        label="Gravando com reprodução (começa quando o vídeo tocar)",
        videoUrl=video_uri,
        cues=cues,
        vttText=vtt_text,
        autoStartOnPlay=True,
        key="karaoke_player",
    )

    if isinstance(result, dict) and result.get("status") == "recorded" and result.get("b64"):
        try:
            b64_str = result["b64"].split(",")[-1]
            st.session_state["recorded_singing"] = base64.b64decode(b64_str)
            st.session_state["recorded_singing_trigger"] = result.get("trigger", "manual")
            st.session_state["recorded_singing_version"] = time.time()
            add_log(f"Gravação de voz capturada durante a reprodução (gatilho={result.get('trigger', 'manual')})")
        except Exception as exc:
            add_log(f"Não foi possível decodificar a gravação da voz: {exc}")


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


def _segment_audio(source) -> "AudioSegment":
    """Load an audio segment from path, file-like, or bytes."""
    from pydub import AudioSegment  # lazy import

    if isinstance(source, Path):
        return AudioSegment.from_file(source)
    if isinstance(source, (bytes, bytearray)):
        return AudioSegment.from_file(io.BytesIO(source))
    if hasattr(source, "read"):
        data = source.read()
        return AudioSegment.from_file(io.BytesIO(data))
    raise ValueError("Fonte de áudio não suportada")


def _audio_to_vector(segment, frame_ms: int = 500) -> list[float]:
    """Convert audio to a coarse energy vector for similarity comparison."""
    # Normalize format
    seg = segment.set_channels(1).set_frame_rate(16000)
    samples = seg.get_array_of_samples()
    frame_len = max(int(seg.frame_rate * frame_ms / 1000), 1)
    vec = []
    acc = 0
    count = 0
    for s in samples:
        acc += abs(int(s))
        count += 1
        if count >= frame_len:
            vec.append(acc / count)
            acc = 0
            count = 0
    if count:
        vec.append(acc / max(count, 1))
    # Normalize vector magnitude to reduce loudness impact
    if not vec:
        return [0.0]
    mean = sum(vec) / len(vec)
    centered = [v - mean for v in vec]
    norm = math.sqrt(sum(v * v for v in centered)) or 1.0
    return [v / norm for v in centered]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    n = min(len(vec_a), len(vec_b))
    a = vec_a[:n]
    b = vec_b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(min(dot / (na * nb), 1.0), -1.0)


@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """Load a transformer encoder for audio embeddings."""
    try:
        from transformers import AutoProcessor, AutoModel  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Dependência ausente: instale transformers e torch para gerar embeddings.") from exc

    model_name = "microsoft/wavlm-base-plus"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _audio_to_embedding(segment) -> np.ndarray:
    """Convert an audio segment to a transformer embedding (mean pooled)."""
    seg = segment.set_channels(1).set_frame_rate(16000)
    samples = np.asarray(seg.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return np.zeros((768,), dtype=np.float32)
    # Normalize to [-1,1]
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Dependência ausente: instale torch para gerar embeddings.") from exc

    processor, model = _load_embedding_model()
    with torch.no_grad():
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (1, T, D)
        pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return pooled


def _segment_to_samples(segment, target_rate: int = 16000) -> np.ndarray:
    """Convert an AudioSegment to mono, normalized float samples."""
    seg = segment.set_channels(1).set_frame_rate(target_rate)
    samples = np.asarray(seg.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return np.zeros((0,), dtype=np.float32)
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    return samples


def _stft_magnitude(samples: np.ndarray, fft_size: int = 2048, hop: int = 512) -> np.ndarray:
    """Compute magnitude spectrogram normalized per frame."""
    if samples.size == 0:
        return np.zeros((1, fft_size // 2 + 1), dtype=np.float32)

    window = np.hanning(fft_size).astype(np.float32)
    frames = []
    for start in range(0, len(samples), hop):
        frame = samples[start : start + fft_size]
        if frame.size == 0:
            break
        if frame.shape[0] < fft_size:
            frame = np.pad(frame, (0, fft_size - frame.shape[0]))
        spectrum = np.fft.rfft(frame * window)
        frames.append(np.abs(spectrum))
        if start + fft_size >= len(samples):
            break

    if not frames:
        frame = np.pad(samples, (0, max(0, fft_size - samples.shape[0])))[:fft_size]
        spectrum = np.fft.rfft(frame * window)
        frames.append(np.abs(spectrum))

    mags = np.vstack(frames)
    norms = np.linalg.norm(mags, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mags / norms


def _fft_similarity_score(reference_seg, user_seg) -> tuple[float, float]:
    """
    Compare two audio segments using spectral similarity derived from FFT magnitudes.
    Returns (score_0_100, similarity_0_1).
    """
    ref_samples = _segment_to_samples(reference_seg)
    user_samples = _segment_to_samples(user_seg)
    if ref_samples.size == 0 or user_samples.size == 0:
        return 0.0, 0.0

    max_len = min(ref_samples.shape[0], user_samples.shape[0])
    if max_len <= 0:
        return 0.0, 0.0
    ref_samples = ref_samples[:max_len]
    user_samples = user_samples[:max_len]

    ref_spec = _stft_magnitude(ref_samples)
    user_spec = _stft_magnitude(user_samples)
    frames = min(ref_spec.shape[0], user_spec.shape[0])
    if frames == 0:
        return 0.0, 0.0
    ref_spec = ref_spec[:frames]
    user_spec = user_spec[:frames]

    sims = np.sum(ref_spec * user_spec, axis=1)
    sims = np.clip(sims, 0.0, 1.0)
    spectral_similarity = float(np.mean(sims))
    score = float(max(0.0, min(spectral_similarity * 100.0, 100.0)))
    return score, spectral_similarity


def _cues_to_timed_words(cues: list[dict]) -> list[tuple[str, float]]:
    words = []
    for cue in cues:
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", start))
        tokens = text.split()
        duration = max(end - start, 0.01)
        step = duration / max(len(tokens), 1)
        for idx, tok in enumerate(tokens):
            words.append((tok.lower(), start + idx * step))
    return words


def _transcribe_with_whisper(audio_segment) -> list[tuple[str, float]]:
    """Use Whisper (transformers pipeline already loaded) to get word + start time."""
    asr = load_asr_pipeline(get_asr_model_name())
    seg = audio_segment.set_channels(1).set_frame_rate(16000)
    samples = np.asarray(seg.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return []
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    result = asr(samples, return_timestamps="word")
    words = []
    for ch in result.get("chunks") or []:
        w = (ch.get("text") or "").strip()
        ts = ch.get("timestamp") or (None, None)
        t0 = ts[0] if isinstance(ts, (list, tuple)) and len(ts) else None
        if w and t0 is not None:
            words.append((w.lower(), float(t0)))
    # Fallback if no chunks
    if not words and result.get("text"):
        words.append((result["text"].strip().lower(), 0.0))
    return words


def _timed_alignment_score(expected: list[tuple[str, float]], observed: list[tuple[str, float]]):
    """Match word order with time offsets; return similarity 0-1."""
    if not expected or not observed:
        return 0.0, 0.0, 0
    exp_words = [w for w, _ in expected]
    obs_words = [w for w, _ in observed]
    matcher = difflib.SequenceMatcher(None, exp_words, obs_words)
    blocks = matcher.get_matching_blocks()

    matched = 0
    deltas = []
    for block in blocks:
        a0, b0, size = block
        for i in range(size):
            exp_idx = a0 + i
            obs_idx = b0 + i
            if exp_idx < len(expected) and obs_idx < len(observed):
                matched += 1
                deltas.append(abs(expected[exp_idx][1] - observed[obs_idx][1]))

    coverage = matched / max(len(exp_words), 1)
    if deltas:
        avg_dt = sum(deltas) / len(deltas)
        # penalize offsets above 2s heavily
        timing = max(0.0, 1.0 - (avg_dt / 2.0))
    else:
        timing = 0.0
    score = coverage * timing
    return score, coverage, timing


def score_singing(song_path: Path, user_audio) -> tuple[float, float]:
    """
    Score by comparing spectral content of the original audio and the user's singing.
    Uses an FFT-based magnitude comparison to produce (score_0_100, spectral_similarity_0_1).
    """
    add_log("Pontuando a voz (similaridade espectral via FFT)...")
    reference_seg = _segment_audio(song_path)
    user_seg = _segment_audio(user_audio)
    score, spectral_similarity = _fft_similarity_score(reference_seg, user_seg)
    add_log(f"Pontuação FFT={score:.1f} (sobreposição espectral={spectral_similarity:.3f})")
    return score, spectral_similarity


def srt_to_webvtt(srt_text: str) -> str:
    """Convert SRT cues into WebVTT format for native <track> captions."""
    cues = parse_srt_cues(srt_text)

    def to_vtt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(",", ".")

    lines = ["WEBVTT", ""]
    for cue in cues:
        start = to_vtt_timestamp(cue.get("start", 0.0))
        end = to_vtt_timestamp(cue.get("end", cue.get("start", 0.0) + 1.0))
        text = (cue.get("text") or "").strip() or "♪"
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def record_audio_component(label: str, key: str) -> Optional[bytes]:
    """
    Render an in-browser recorder component. Returns WAV bytes once recording stops.
    """
    value = karaoke_recorder_component(label=label, key=key, default={"status": "idle"})
    if isinstance(value, dict) and value.get("status") == "recorded" and value.get("b64"):
        try:
            b64_str = value["b64"].split(",")[-1]
            payload = base64.b64decode(b64_str)
            trigger = value.get("trigger", "manual")
            return payload, trigger
        except Exception as exc:  # pragma: no cover
            add_log(f"Não foi possível decodificar a gravação: {exc}")
    return None




def _ensure_subtitles(video_path: str, stepper: Optional[StepProgress] = None):
    """Generate subtitles for the provided video path if not already cached."""
    model_name = get_asr_model_name()
    subtitle_dir = str(get_subtitle_temp_dir())
    if (
        st.session_state.get("srt_video_path") == video_path
        and st.session_state.get("srt_text")
        and st.session_state.get("srt_model_name") == model_name
        and st.session_state.get("srt_dir") == subtitle_dir
        and Path(st.session_state.get("srt_path", "")).exists()
    ):
        if stepper:
            stepper.update(1, 1.0, text="Etapa 2/3: Legendas já em cache")
            stepper.update(2, 1.0, text="Etapa 3/3: Pronto para reproduzir com legendas")
        add_log("Legendas carregadas do cache")
        return

    if stepper:
        stepper.update(1, 0.0, text="Etapa 2/3: Preparando transcrição...")
    add_log("Transcrevendo áudio para gerar legendas...")
    try:
        if stepper:
            stepper.update(1, 0.3, text="Etapa 2/3: Filtrando áudio para voz...")
        srt_text, srt_path = generate_subtitles(video_path, model_name, subtitle_dir)
        if stepper:
            stepper.update(1, 1.0, text="Etapa 2/3: Legendas prontas")
    except Exception as exc:
        err = f"Não foi possível gerar legendas: {exc}"
        add_log(err)
        st.error(err)
    else:
        st.session_state["srt_text"] = srt_text
        st.session_state["srt_path"] = srt_path
        st.session_state["srt_video_path"] = video_path
        st.session_state["srt_model_name"] = model_name
        st.session_state["srt_dir"] = subtitle_dir
        add_log(f"Legendas salvas em {srt_path}")
        if stepper:
            stepper.update(2, 1.0, text="Etapa 3/3: Pronto para reproduzir com legendas")


def auto_score_if_ready():
    version = st.session_state.get("recorded_singing_version")
    last_scored = st.session_state.get("recorded_singing_scored_version")
    video_path_val = st.session_state.get("video_path")
    if not video_path_val:
        return
    path_obj = Path(video_path_val)
    if not path_obj.exists():
        add_log("Pontuação automática ignorada: arquivo de vídeo não encontrado.")
        return
    recording = st.session_state.get("recorded_singing")
    if recording is None:
        return

    if version and (last_scored is None or version > last_scored):
        try:
            with st.spinner("Pontuando sua voz..."):
                score, sim_raw = score_singing(path_obj, recording)
        except Exception as exc:
            msg = f"Pontuação automática falhou: {exc}"
            add_log(msg)
        else:
            st.session_state["last_score"] = score
            st.session_state["last_spectral_similarity"] = sim_raw
            st.session_state["recorded_singing_scored_version"] = version
            add_log(f"Voz pontuada automaticamente (gatilho={st.session_state.get('recorded_singing_trigger', 'auto')}): {score:.1f}")


def render_karaoke_page():
    st.title("Carregador de vídeos do YouTube")
    st.write(
        "Cole a URL do YouTube, baixe para uma pasta temporária e reproduza aqui mesmo."
    )

    url = st.text_input(
        "URL do YouTube",
        placeholder="https://www.youtube.com/watch?v=...",
        value="https://www.youtube.com/watch?v=8AHCfZTRGiI&list=RD8AHCfZTRGiI&start_radio=1",
    )

    if st.button("Baixar e mostrar"):
        if not url.strip():
            add_log("Forneça um link válido do YouTube.")
        else:
            progress_bar = st.progress(0, text="Etapa 1/3: Iniciando download...")
            stepper = StepProgress(progress_bar, steps=3)
            stepper.update(0, 0.0, text="Etapa 1/3: Iniciando download...")
            try:
                video_path = download_video(url.strip(), progress_callback=_progress_hook(stepper, 0))
            except Exception as exc:
                err = f"Não foi possível baixar o vídeo: {exc}"
                add_log(err)
                st.error(err)
            else:
                stepper.update(0, 1.0, text="Etapa 1/3: Download concluído")
                st.session_state["video_path"] = video_path
                add_log(f"Baixado em {video_path}")
                _ensure_subtitles(video_path, stepper)
            # leave progress bar visible to show completion state

    if "video_path" in st.session_state:
        path = Path(st.session_state["video_path"])
        if path.exists():
            if (
                st.session_state.get("srt_video_path") != str(path)
                or st.session_state.get("srt_model_name") != get_asr_model_name()
                or st.session_state.get("srt_dir") != str(get_subtitle_temp_dir())
            ):
                progress_bar = st.progress(0, text="Etapa 1/3: Usando download existente...")
                stepper = StepProgress(progress_bar, steps=3)
                stepper.update(0, 1.0, text="Etapa 1/3: Usando download existente")
                _ensure_subtitles(str(path), stepper)
            else:
                stepper = None

            if "srt_text" in st.session_state:
                render_video_with_subtitles(path, st.session_state["srt_text"])
            else:
                st.video(str(path))
            if st.session_state.get("last_play_logged") != str(path):
                add_log(f"Reproduzindo de: {path}")
                st.session_state["last_play_logged"] = str(path)
        else:
            add_log("Arquivo baixado não encontrado. Tente baixar novamente.")

    st.divider()
    st.subheader("Pontuação do karaokê")
    st.caption("Grave sua voz enquanto a música toca. A pontuação usa similaridade espectral por FFT.")

    auto_score_if_ready()

    if st.button("Pontuar minha voz"):
        if "video_path" not in st.session_state or not Path(st.session_state["video_path"]).exists():
            msg = "Nenhuma música baixada encontrada. Baixe um vídeo antes."
            add_log(msg)
            st.error(msg)
        elif "recorded_singing" not in st.session_state or st.session_state.get("recorded_singing") is None:
            msg = "Toque o vídeo para gravar sua voz primeiro."
            add_log(msg)
            st.error(msg)
        else:
            try:
                with st.spinner("Pontuando sua voz..."):
                    score, sim_raw = score_singing(
                        Path(st.session_state["video_path"]),
                        st.session_state["recorded_singing"],
                    )
            except Exception as exc:
                msg = f"Não foi possível pontuar a voz: {exc}"
                add_log(msg)
                st.error(msg)
            else:
                st.session_state["last_score"] = score
                st.session_state["last_spectral_similarity"] = sim_raw
                st.session_state["recorded_singing_scored_version"] = st.session_state.get("recorded_singing_version")

    if st.session_state.get("last_score") is not None:
        st.metric("Pontuação por FFT", f"{st.session_state['last_score']:.1f} / 100")
        st.caption(f"sobreposição espectral={st.session_state.get('last_spectral_similarity', 0):.3f}")

    render_terminal()


def render_settings_page():
    st.title("Configurações")
    st.caption("Escolha onde os arquivos serão guardados e qual modelo de fala será usado.")

    model_options = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
    ]
    current_model = get_asr_model_name()
    if current_model not in model_options:
        model_options = [current_model] + model_options
    try:
        model_idx = model_options.index(current_model)
    except ValueError:
        model_idx = 0

    with st.form("settings_form", clear_on_submit=False):
        video_dir_input = st.text_input(
            "Pasta temporária para vídeos baixados",
            value=st.session_state.get("video_temp_dir", str(DEFAULT_MEDIA_ROOT)),
            help="Novos downloads serão salvos aqui.",
        )
        subtitle_dir_input = st.text_input(
            "Pasta temporária para legendas geradas",
            value=st.session_state.get("subtitle_temp_dir", st.session_state.get("video_temp_dir", str(DEFAULT_MEDIA_ROOT))),
            help="Arquivos SRT serão escritos aqui.",
        )
        model_choice = st.selectbox(
            "Modelo de transcrição (voz para texto)",
            options=model_options,
            index=model_idx,
            help="Modelos maiores são mais lentos, mas podem melhorar a qualidade da transcrição.",
        )
        submitted = st.form_submit_button("Salvar configurações")

    if submitted:
        st.session_state["video_temp_dir"] = video_dir_input.strip() or str(DEFAULT_MEDIA_ROOT)
        st.session_state["subtitle_temp_dir"] = subtitle_dir_input.strip() or st.session_state["video_temp_dir"]
        st.session_state["asr_model_name"] = model_choice
        video_dir = get_video_temp_dir()
        subtitle_dir = get_subtitle_temp_dir()
        add_log(f"Configurações atualizadas: vídeos em {video_dir}, legendas em {subtitle_dir}, modelo={model_choice}")
        st.success("Configurações salvas. Novos downloads/transcrições usarão esses locais.")
        st.info(
            f"Vídeos -> {video_dir}\nLegendas -> {subtitle_dir}\nModelo -> {model_choice}",
            icon="ℹ️",
        )

    st.markdown("**Configuração atual**")
    st.code(
        f"Vídeos: {get_video_temp_dir()}\nLegendas: {get_subtitle_temp_dir()}\nModelo: {get_asr_model_name()}",
        language="text",
    )


def run_app():
    page = st.sidebar.radio("Navegação", ["Karaokê", "Configurações"], index=0)
    with st.sidebar:
        st.caption(f"Vídeos: {get_video_temp_dir()}")
        st.caption(f"Legendas: {get_subtitle_temp_dir()}")
        st.caption(f"Modelo: {get_asr_model_name()}")

    if page == "Karaokê":
        render_karaoke_page()
    else:
        render_settings_page()


if STREAMLIT_ACTIVE:
    run_app()
