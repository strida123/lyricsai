import streamlit as st
import requests
import tempfile
import os
import srt
import zipfile
import math
import subprocess
from datetime import timedelta

########################################
# 1) Save uploaded file preserving original extension
########################################

def save_temp_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, dest_suffix: str) -> str:
    """Write uploaded_file bytes to a NamedTemporaryFile and return its path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=dest_suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name

########################################
# 2) Demucs Pre‑processing (optional)
########################################

def run_demucs_vocals_mp3(input_audio_path: str, two_stems: bool = True) -> str:
    """Run Demucs stem separation and return path to vocals.mp3."""
    output_dir = tempfile.mkdtemp()

    cmd_demucs = [
        "demucs",
        input_audio_path,
        "-o",
        output_dir,
    ]
    if two_stems:
        cmd_demucs += ["--two-stems", "vocals"]

    subprocess.run(cmd_demucs, check=True)

    # Locate vocals.wav
    vocals_wav_path = None
    for root, _, files in os.walk(output_dir):
        if "vocals.wav" in files:
            vocals_wav_path = os.path.join(root, "vocals.wav")
            break

    if not vocals_wav_path:
        raise FileNotFoundError("vocals.wav not found after Demucs run.")

    vocals_mp3_path = vocals_wav_path.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", vocals_wav_path, vocals_mp3_path], check=True)
    return vocals_mp3_path

########################################
# 3) Build ZIP helper
########################################

def create_zip(audio_basename: str, original_srt: str, aligned_srt: str) -> str:
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{audio_basename}_transcription.srt", original_srt)
        zf.writestr(f"{audio_basename}_corrected_transcription.srt", aligned_srt)
    return zip_path

########################################
# 4) Convert SRT -> per‑word timestamps (uniform split)
########################################

def srt_to_word_timestamps(srt_text: str):
    subs = list(srt.parse(srt_text))
    word_entries = []
    for sub in subs:
        words = sub.content.strip().split()
        if not words:
            continue
        total_dur = (sub.end - sub.start).total_seconds()
        dur_per_word = total_dur / len(words)
        for i, w in enumerate(words):
            w_start = sub.start + timedelta(seconds=i * dur_per_word)
            w_end = sub.start + timedelta(seconds=(i + 1) * dur_per_word)
            word_entries.append((w, w_start, w_end))
    return word_entries

########################################
# 5) Lyrics helpers
########################################

def lyrics_to_word_list(lyrics_text: str):
    lines = lyrics_text.split("\n")
    word_list = []
    for idx, line in enumerate(lines):
        for w in line.strip().split():
            word_list.append((w, idx))
    return word_list, lines


def normalize_word(word: str) -> str:
    w = word.lower().replace("-", " ")
    return "".join(ch for ch in w if ch.isalnum() or ch.isspace()).strip()

########################################
# 6) Dynamic‑programming word alignment
########################################

def word_alignment(transcript_words, lyric_words):
    T, L = len(transcript_words), len(lyric_words)
    dp = [[(math.inf, None) for _ in range(T + 1)] for _ in range(L + 1)]
    dp[0][0] = (0, None)
    skip_penalty = 1

    def cost(lw, tw):
        return 0 if normalize_word(lw) == normalize_word(tw) else 1

    for i in range(L + 1):
        for j in range(T + 1):
            curr_cost, _ = dp[i][j]
            if curr_cost == math.inf:
                continue
            if i < L and curr_cost + skip_penalty < dp[i + 1][j][0]:
                dp[i + 1][j] = (curr_cost + skip_penalty, (i, j))
            if j < T and curr_cost + skip_penalty < dp[i][j + 1][0]:
                dp[i][j + 1] = (curr_cost + skip_penalty, (i, j))
            if i < L and j < T:
                m_cost = curr_cost + cost(lyric_words[i][0], transcript_words[j][0])
                if m_cost < dp[i + 1][j + 1][0]:
                    dp[i + 1][j + 1] = (m_cost, (i, j))

    alignment = [None] * L
    i, j = L, T
    while i > 0 or j > 0:
        _, back = dp[i][j]
        if back is None:
            break
        pi, pj = back
        if pi == i - 1 and pj == j - 1:
            alignment[i - 1] = j - 1
        i, j = pi, pj
    return alignment

########################################
# 7) Build corrected SRT from alignment
########################################

def build_lyrics_srt_from_word_alignment(lyrics_text, alignment, transcript_words):
    lines = [l.strip() for l in lyrics_text.split("\n")]
    lyric_word_list, _ = lyrics_to_word_list(lyrics_text)
    line_to_t_indices = {}
    for i, (_, line_idx) in enumerate(lyric_word_list):
        if alignment[i] is not None:
            t_idx = alignment[i]
            line_to_t_indices.setdefault(line_idx, []).append(t_idx)

    srt_entries, counter = [], 1
    for line_idx, text in enumerate(lines):
        idxs = line_to_t_indices.get(line_idx, [])
        if not idxs:
            continue
        start = min(transcript_words[k][1] for k in idxs)
        end = max(transcript_words[k][2] for k in idxs)
        srt_entries.append(srt.Subtitle(index=counter, start=start, end=end, content=text))
        counter += 1
    srt_entries.sort(key=lambda s: s.start)
    return srt.compose(srt_entries)

########################################
# 8) Main alignment wrapper
########################################

def lyrics_driven_srt(whisper_srt: str, lyrics: str):
    transcript_word_list = srt_to_word_timestamps(whisper_srt)
    lyric_word_list, _ = lyrics_to_word_list(lyrics)
    alignment = word_alignment(transcript_word_list, lyric_word_list)
    return build_lyrics_srt_from_word_alignment(lyrics, alignment, transcript_word_list)

########################################
# Streamlit App – updated for GPT‑4o‑Transcribe
########################################

st.set_page_config(layout="wide")
st.title("TVG LyricsAI + Demucs  |  Powered by GPT‑4o‑Transcribe")

if "vocals_bytes" not in st.session_state:
    st.session_state.vocals_bytes = None
if "srt_zip_bytes" not in st.session_state:
    st.session_state.srt_zip_bytes = None

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Use Demucs vocal isolation?")
    stems_choice = None
    if use_demucs:
        stems_choice = st.selectbox("Number of stems", ["2 stems (vocals)", "4 stems"])
    st.markdown("[Get an API key](https://platform.openai.com/account/api-keys)")

uploaded_audio = st.file_uploader(
    "Upload audio (max 25 MB per OpenAI limits)",
    type=["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"],
)
lyrics_input = st.text_area("Paste lyrics (optional)")

if uploaded_audio:
    if not openai_api_key:
        st.warning("Add your OpenAI key in the sidebar.")
        st.stop()

    if st.button("Generate SRT"):
        st.session_state.vocals_bytes = None
        st.session_state.srt_zip_bytes = None

        # Preserve original extension & mime‑type
        file_ext = os.path.splitext(uploaded_audio.name)[1]
        audio_path = save_temp_file(uploaded_audio, file_ext)
        audio_basename = os.path.splitext(uploaded_audio.name)[0]
        mime_type = uploaded_audio.type or "application/octet-stream"

        # Optional Demucs isolation
        if use_demucs:
            two_stems = not (stems_choice and stems_choice.startswith("4"))
            with st.spinner("Running Demucs …"):
                vocals_mp3 = run_demucs_vocals_mp3(audio_path, two_stems=two_stems)
            with open(vocals_mp3, "rb") as vf:
                st.session_state.vocals_bytes = vf.read()
            processed_path = vocals_mp3
            mime_type = "audio/mpeg"
        else:
            processed_path = audio_path

        ############################################################
        # 3) Transcribe with GPT‑4o‑Transcribe
        ############################################################
        transcribe_url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        data = {
            "model": "gpt-4o-transcribe",
            "response_format": "srt",
            "temperature": 0,
        }

        with st.spinner("Transcribing with GPT‑4o …"):
            with open(processed_path, "rb") as f:
                files = {"file": (os.path.basename(processed_path), f, mime_type)}
                resp = requests.post(transcribe_url, headers=headers, data=data, files=files, timeout=600)
                resp.raise_for_status()
            original_srt = resp.text

        # Optional lyric alignment
        if lyrics_input.strip():
            with st.spinner("Aligning to provided lyrics …"):
                final_srt = lyrics_driven_srt(original_srt, lyrics_input)
        else:
            final_srt = original_srt

        # Package ZIP
        zip_path = create_zip(audio_basename, original_srt, final_srt)
        with open(zip_path, "rb") as zf:
            st.session_state.srt_zip_bytes = zf.read()

        st.success("Done! Download your files below.")

# Download buttons (persist across reruns)
if st.session_state.vocals_bytes:
    st.download_button(
        "Download isolated vocals (MP3)",
        data=st.session_state.vocals_bytes,
        file_name="vocals.mp3",
        mime="audio/mpeg",
    )

if st.session_state.srt_zip_bytes:
    st.download_button(
        "Download SRT files (ZIP)",
        data=st.session_state.srt_zip_bytes,
        file_name="srt_files.zip",
        mime="application/zip",
    )
