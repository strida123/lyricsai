# lyricsai_gpt4o.py  –  TVG LyricsAI + Demucs (GPT-4o Edition)
# -----------------------------------------------------------
# Features
#   • Optional Demucs: 2-stem or 4-stem isolation
#   • Transcription with gpt-4o-transcribe (word timestamps requested)
#   • Optional lyrics alignment → corrected SRT
#   • ZIP packaging of original & corrected subtitles
#   • Persistent download buttons; fatal errors surface in-app

import streamlit as st
import requests
import json
import tempfile
import os
import srt
import zipfile
import math
import subprocess
from datetime import timedelta
from contextlib import contextmanager

# ─────────────────────────  Convenience: error wrapper  ──────────────────────
@contextmanager
def safe_step(label: str):
    """
    Show a Streamlit spinner; on exception display error and halt app.
    """
    try:
        with st.spinner(label):
            yield
    except Exception as e:
        st.error(f"{label} failed → {e}")
        st.stop()

# ─────────────────────────  File helpers  ─────────────────────────────────────
def save_temp_file(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name

# ─────────────────────────  Demucs isolation  ────────────────────────────────
def run_demucs_vocals_mp3(input_path: str, two_stems: bool = True) -> str:
    """
    Run Demucs on *input_path* and return the path to vocals.mp3.
    Requires `demucs` CLI and `ffmpeg` on PATH.
    """
    out_dir = tempfile.mkdtemp()
    cmd = ["demucs", input_path, "-o", out_dir]
    if two_stems:
        cmd += ["--two-stems", "vocals"]
    subprocess.run(cmd, check=True)

    vocals_wav = None
    for root, _, files in os.walk(out_dir):
        if "vocals.wav" in files:
            vocals_wav = os.path.join(root, "vocals.wav")
            break
    if vocals_wav is None:
        raise FileNotFoundError("vocals.wav not produced by Demucs")

    vocals_mp3 = vocals_wav.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, vocals_mp3], check=True)
    return vocals_mp3

# ─────────────────────────  ZIP helper  ───────────────────────────────────────
def create_zip(basename: str, original_srt: str, aligned_srt: str) -> str:
    zp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(f"{basename}_transcription.srt", original_srt)
        z.writestr(f"{basename}_corrected_transcription.srt", aligned_srt)
    return zp

# ─────────────────────────  OpenAI JSON → SRT & word list  ───────────────────
def json_to_srt_and_words(json_str: str):
    data = json.loads(json_str)
    subs, words, idx = [], [], 1
    for seg in data.get("segments", []):
        start = timedelta(seconds=seg["start"])
        end   = timedelta(seconds=seg["end"])
        text  = seg["text"].strip()
        subs.append(srt.Subtitle(idx, start, end, text))
        idx += 1
        for w in seg.get("words", []):
            words.append((
                w["word"],
                timedelta(seconds=w["start"]),
                timedelta(seconds=w["end"]),
            ))

    # Fallback – fabricate per-word timings if API omitted them
    if not words:
        for sub in subs:
            tokens = sub.content.split()
            dur = (sub.end - sub.start).total_seconds() / max(len(tokens), 1)
            for i, tok in enumerate(tokens):
                words.append((
                    tok,
                    sub.start + timedelta(seconds=i * dur),
                    sub.start + timedelta(seconds=(i + 1) * dur),
                ))

    return srt.compose(subs), words

# ─────────────────────────  Lyrics-alignment helpers  ────────────────────────
def normalize_word(word: str) -> str:
    return "".join(ch for ch in word.lower().replace("-", " ")
                   if ch.isalnum() or ch.isspace()).strip()

def lyrics_to_word_list(text: str):
    lines = text.splitlines()
    out = []
    for li, line in enumerate(lines):
        for w in line.strip().split():
            out.append((w, li))
    return out, lines

def word_alignment(trans_words, lyric_words):
    T, L, skip = len(trans_words), len(lyric_words), 1
    dp = [[(math.inf, None) for _ in range(T + 1)] for _ in range(L + 1)]
    dp[0][0] = (0, None)

    for i in range(L + 1):
        for j in range(T + 1):
            cost, _ = dp[i][j]
            if cost == math.inf:
                continue
            # skip lyric
            if i < L and cost + skip < dp[i + 1][j][0]:
                dp[i + 1][j] = (cost + skip, (i, j))
            # skip transcript
            if j < T and cost + skip < dp[i][j + 1][0]:
                dp[i][j + 1] = (cost + skip, (i, j))
            # match
            if i < L and j < T:
                add = 0 if normalize_word(lyric_words[i][0]) == \
                          normalize_word(trans_words[j][0]) else 1
                if cost + add < dp[i + 1][j + 1][0]:
                    dp[i + 1][j + 1] = (cost + add, (i, j))

    align = [None] * L
    i, j = L, T
    while i or j:
        _, back = dp[i][j]
        if back is None:
            break
        pi, pj = back
        if pi == i - 1 and pj == j - 1:
            align[i - 1] = j - 1
        i, j = pi, pj
    return align

def build_lyrics_srt(lyrics_text, align, trans_words):
    lines = [l.strip() for l in lyrics_text.splitlines()]
    lyric_words, _ = lyrics_to_word_list(lyrics_text)
    line_map = {}
    for idx, (_, li) in enumerate(lyric_words):
        if align[idx] is not None:
            line_map.setdefault(li, []).append(align[idx])

    subs, counter = [], 1
    for li, text in enumerate(lines):
        idxs = line_map.get(li, [])
        if not idxs:
            continue
        start = min(trans_words[k][1] for k in idxs)
        end   = max(trans_words[k][2] for k in idxs)
        subs.append(srt.Subtitle(counter, start, end, text))
        counter += 1
    subs.sort(key=lambda s: s.start)
    return srt.compose(subs)

def align_lyrics(trans_words, lyrics_text):
    lyric_words, _ = lyrics_to_word_list(lyrics_text)
    align = word_alignment(trans_words, lyric_words)
    return build_lyrics_srt(lyrics_text, align, trans_words)

# ─────────────────────────  Streamlit UI  ────────────────────────────────────
st.set_page_config(layout="wide")
st.title("TVG LyricsAI + Demucs  |  Powered by GPT-4o-Transcribe")

# Persistent session blobs
for blob in ("vocals_bytes", "zip_bytes"):
    st.session_state.setdefault(blob, None)

# Sidebar
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Isolate vocals with Demucs?")
    stems_choice = (st.selectbox("Demucs stems",
                                 ["2 stems (vocals)", "4 stems"],
                                 disabled=not use_demucs)
                    if use_demucs else None)
    st.markdown("[Need an API key?](https://platform.openai.com/account/api-keys)")

# Main controls
uploaded_audio = st.file_uploader(
    "Upload audio (≤25 MB per OpenAI limit)",
    type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"]
)
lyrics_input = st.text_area("Paste lyrics (optional)")

if uploaded_audio and st.button("Generate SRT"):
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    # Clear previous results
    st.session_state.vocals_bytes = None
    st.session_state.zip_bytes = None

    # Save upload
    file_ext = os.path.splitext(uploaded_audio.name)[1] or ".wav"
    audio_path = save_temp_file(uploaded_audio, file_ext)
    base_name = os.path.splitext(uploaded_audio.name)[0]

    # Optional Demucs
    proc_path, mime = audio_path, (uploaded_audio.type or "audio/wav")
    if use_demucs:
        two_stems = not (stems_choice and stems_choice.startswith("4"))
        with safe_step("Running Demucs"):
            vocals_mp3 = run_demucs_vocals_mp3(audio_path, two_stems=two_stems)
        if os.path.getsize(vocals_mp3) < 1_000_000:
            raise RuntimeError("Demucs produced an unusually small MP3")
        with open(vocals_mp3, "rb") as vf:
            st.session_state.vocals_bytes = vf.read()
        proc_path, mime = vocals_mp3, "audio/mpeg"

    # Transcribe
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "timestamp_granularities[]": "word",
        "temperature": [0, 0.2, 0.4],
    }
    with safe_step("Transcribing with GPT-4o"):
        with open(proc_path, "rb") as f:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": (os.path.basename(proc_path), f, mime)},
                data=data,
                timeout=600,
            )
        resp.raise_for_status()

    original_srt, transcript_words = json_to_srt_and_words(resp.text)

    # Lyrics alignment
    if lyrics_input.strip():
        with safe_step("Aligning to provided lyrics"):
            final_srt = align_lyrics(transcript_words, lyrics_input)
    else:
        final_srt = original_srt

    # Package ZIP
    with safe_step("Packaging ZIP"):
        zip_path = create_zip(base_name, original_srt, final_srt)
        with open(zip_path, "rb") as zf:
            st.session_state.zip_bytes = zf.read()

    st.success("Done – download below!")

# ─────────────────────────  Download buttons  ────────────────────────────────
if st.session_state.vocals_bytes:
    st.download_button(
        "Download isolated vocals (MP3)",
        st.session_state.vocals_bytes,
        "vocals.mp3",
        mime="audio/mpeg",
    )
elif uploaded_audio and use_demucs:
    st.info("No vocals MP3 – Demucs failed or not run yet.")

if st.session_state.zip_bytes:
    st.download_button(
        "Download captions (ZIP)",
        st.session_state.zip_bytes,
        "captions.zip",
        mime="application/zip",
    )
elif uploaded_audio:
    st.info("No captions ZIP yet.")
