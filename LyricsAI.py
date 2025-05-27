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

# ─────────────────────────  Utility: temp files  ────────────────────────────
def save_temp_file(uploaded_file, suffix):
    """Save a Streamlit UploadedFile to a temp file and return its path."""
    fn = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fn.write(uploaded_file.getvalue())
    fn.close()
    return fn.name

# ─────────────────────────  Demucs isolation  ───────────────────────────────
def run_demucs_vocals_mp3(input_path, two_stems=True):
    """
    Run Demucs on *input_path* and return vocals.mp3 path.
    If *two_stems* is False, Demucs uses its default 4-stem separation.
    """
    out_dir = tempfile.mkdtemp()
    cmd = ["demucs", input_path, "-o", out_dir]
    if two_stems:
        cmd += ["--two-stems", "vocals"]
    subprocess.run(cmd, check=True)

    # Locate vocals.wav produced by Demucs
    vocals_wav = None
    for root, _, files in os.walk(out_dir):
        if "vocals.wav" in files:
            vocals_wav = os.path.join(root, "vocals.wav")
            break
    if vocals_wav is None:
        raise FileNotFoundError("vocals.wav not found after Demucs run.")

    vocals_mp3 = vocals_wav.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, vocals_mp3], check=True)
    return vocals_mp3

# ─────────────────────────  ZIP helper  ──────────────────────────────────────
def create_zip(basename, original_srt, aligned_srt):
    """Return path to a zip containing the two SRT variants."""
    zpath = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"{basename}_transcription.srt", original_srt)
        zf.writestr(f"{basename}_corrected_transcription.srt", aligned_srt)
    return zpath

# ─────────────────────────  JSON → SRT + word list  ─────────────────────────
def json_to_srt_and_words(json_str):
    obj = json.loads(json_str)
    subs, words, idx = [], [], 1
    for seg in obj.get("segments", []):
        start = timedelta(seconds=seg["start"])
        end   = timedelta(seconds=seg["end"])
        text  = seg["text"].strip()
        subs.append(srt.Subtitle(idx, start, end, text))
        idx += 1
        for w in seg.get("words", []):
            words.append((w["word"],
                          timedelta(seconds=w["start"]),
                          timedelta(seconds=w["end"])))
    # Fallback: fabricate per-word timings if the API didn’t include them
    if not words:
        for s in subs:
            tokens = s.content.split()
            dur = (s.end - s.start).total_seconds() / max(len(tokens), 1)
            for i, tok in enumerate(tokens):
                words.append((tok,
                              s.start + timedelta(seconds=i*dur),
                              s.start + timedelta(seconds=(i+1)*dur)))
    return srt.compose(subs), words

# ─────────────────────────  Lyrics-alignment helpers  ───────────────────────
def normalize_word(w):
    return "".join(ch for ch in w.lower().replace("-", " ")
                   if ch.isalnum() or ch.isspace()).strip()

def lyrics_to_word_list(text):
    lines = text.splitlines()
    out = []
    for li, line in enumerate(lines):
        for w in line.strip().split():
            out.append((w, li))
    return out, lines

def word_alignment(trans_words, lyric_words):
    T, L, skip_pen = len(trans_words), len(lyric_words), 1
    dp = [[(math.inf, None) for _ in range(T+1)] for _ in range(L+1)]
    dp[0][0] = (0, None)
    for i in range(L+1):
        for j in range(T+1):
            cost, _ = dp[i][j]
            if cost == math.inf:
                continue
            # skip lyric
            if i < L and cost + skip_pen < dp[i+1][j][0]:
                dp[i+1][j] = (cost + skip_pen, (i, j))
            # skip transcript
            if j < T and cost + skip_pen < dp[i][j+1][0]:
                dp[i][j+1] = (cost + skip_pen, (i, j))
            # match
            if i < L and j < T:
                extra = 0 if normalize_word(lyric_words[i][0]) == \
                            normalize_word(trans_words[j][0]) else 1
                if cost + extra < dp[i+1][j+1][0]:
                    dp[i+1][j+1] = (cost + extra, (i, j))
    align = [None]*L
    i, j = L, T
    while i or j:
        _, back = dp[i][j]
        if back is None:
            break
        pi, pj = back
        if pi == i-1 and pj == j-1:
            align[i-1] = j-1
        i, j = pi, pj
    return align

def build_lyrics_srt(lyrics_text, align, trans_words):
    lines = [l.strip() for l in lyrics_text.splitlines()]
    lyric_words, _ = lyrics_to_word_list(lyrics_text)
    line_map = {}
    for i, (_, li) in enumerate(lyric_words):
        if align[i] is not None:
            line_map.setdefault(li, []).append(align[i])
    subs, idx = [], 1
    for li, txt in enumerate(lines):
        idxs = line_map.get(li, [])
        if not idxs:
            continue
        start = min(trans_words[k][1] for k in idxs)
        end   = max(trans_words[k][2] for k in idxs)
        subs.append(srt.Subtitle(idx, start, end, txt))
        idx += 1
    subs.sort(key=lambda s: s.start)
    return srt.compose(subs)

def align_lyrics(trans_words, lyrics_text):
    lyric_words, _ = lyrics_to_word_list(lyrics_text)
    align = word_alignment(trans_words, lyric_words)
    return build_lyrics_srt(lyrics_text, align, trans_words)

# ─────────────────────────  Streamlit UI  ───────────────────────────────────
st.set_page_config(layout="wide")
st.title("TVG LyricsAI + Demucs (GPT-4o Edition)")

# Session state for persistent download buttons
for key in ("vocals_bytes", "zip_bytes"):
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar controls
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Isolate vocals with Demucs?")
    stems_choice = st.selectbox("Demucs stems",
                                ["2 stems (vocals)", "4 stems"],
                                disabled=not use_demucs) if use_demucs else None
    st.markdown("[Get an API key](https://platform.openai.com/account/api-keys)")

# Main UI
up_audio = st.file_uploader(
    "Upload Audio",
    type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"]
)
lyrics_txt = st.text_area("Paste lyrics (optional)")

if up_audio and st.button("Generate SRT"):
    if not api_key:
        st.warning("Add your OpenAI key first.")
        st.stop()

    # Reset downloads for a fresh run
    st.session_state.vocals_bytes = None
    st.session_state.zip_bytes = None

    # Save upload
    file_ext = os.path.splitext(up_audio.name)[1] or ".wav"
    audio_path = save_temp_file(up_audio, file_ext)
    base_name = os.path.splitext(up_audio.name)[0]

    # Optional Demucs
    if use_demucs:
        two_stems = not (stems_choice and stems_choice.startswith("4"))
        with st.spinner("Running Demucs …"):
            vocals_mp3 = run_demucs_vocals_mp3(audio_path, two_stems=two_stems)
        with open(vocals_mp3, "rb") as vf:
            st.session_state.vocals_bytes = vf.read()
        proc_path, mime = vocals_mp3, "audio/mpeg"
    else:
        proc_path, mime = audio_path, (up_audio.type or "audio/wav")

    # ──────────  GPT-4o transcription  ──────────
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-transcribe",
        "response_format": "json",
        "timestamp_granularities[]": "word",
        "temperature": 0,
    }
    with st.spinner("Transcribing with GPT-4o …"):
        with open(proc_path, "rb") as f:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": (os.path.basename(proc_path), f, mime)},
                data=data,
                timeout=600,
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            st.error(f"OpenAI error: {resp.text or e}")
            st.stop()

    original_srt, trans_words = json_to_srt_and_words(resp.text)

    # Optional lyric alignment
    if lyrics_txt.strip():
        with st.spinner("Aligning lyrics …"):
            final_srt = align_lyrics(trans_words, lyrics_txt)
    else:
        final_srt = original_srt

    # ZIP packaging
    zip_path = create_zip(base_name, original_srt, final_srt)
    with open(zip_path, "rb") as zf:
        st.session_state.zip_bytes = zf.read()

    st.success("All done! Downloads below ↓")

# ─────────────────────────  Download buttons  ───────────────────────────────
if st.session_state.vocals_bytes:
    st.download_button(
        label="Download isolated vocals (MP3)",
        data=st.session_state.vocals_bytes,
        file_name="vocals.mp3",
        mime="audio/mpeg",
    )

if st.session_state.zip_bytes:
    st.download_button(
        label="Download SRT files (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="captions.zip",
        mime="application/zip",
    )
