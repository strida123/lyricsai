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
# 1) Save uploaded file
########################################
def save_temp_file(uploaded_file, suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name

########################################
# 2) Demucs Preprocessing
########################################
def run_demucs_vocals_mp3(input_audio_path, two_stems=True):
    """
    If two_stems=True => 2-stem (vocals + no_vocals).
    If two_stems=False => 4-stem default (vocals, drums, bass, other).
    Converts the resulting vocals.wav -> vocals.mp3 and returns that path.
    """
    output_dir = tempfile.mkdtemp()

    cmd_demucs = [
        "demucs",
        input_audio_path,
        "-o", output_dir,
    ]
    if two_stems:
        cmd_demucs += ["--two-stems", "vocals"]

    subprocess.run(cmd_demucs, check=True)

    # locate vocals.wav
    vocals_wav_path = None
    for root, dirs, files in os.walk(output_dir):
        if "vocals.wav" in files:
            vocals_wav_path = os.path.join(root, "vocals.wav")
            break

    if not vocals_wav_path:
        raise FileNotFoundError("vocals.wav not found after Demucs.")

    vocals_mp3 = vocals_wav_path.replace(".wav", ".mp3")
    cmd_ffmpeg = ["ffmpeg", "-y", "-i", vocals_wav_path, vocals_mp3]
    subprocess.run(cmd_ffmpeg, check=True)

    return vocals_mp3

########################################
# 3) Make ZIP with original + final
########################################
def create_zip(audio_basename, original_srt_str, aligned_srt_str):
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr(f"{audio_basename}_transcription.srt", original_srt_str)
        zipf.writestr(f"{audio_basename}_corrected_transcription.srt", aligned_srt_str)
    return zip_path

########################################
# 4) Convert Whisper SRT → word timestamps
########################################
def srt_to_word_timestamps(srt_text):
    subs = list(srt.parse(srt_text))
    word_entries = []
    for sub in subs:
        sub_text = sub.content.strip()
        words = sub_text.split()
        if not words:
            continue
        total_sub_duration = (sub.end - sub.start).total_seconds()
        dur_per_word = total_sub_duration / len(words) if len(words) else 0
        for i, w in enumerate(words):
            w_start = sub.start + timedelta(seconds=i * dur_per_word)
            w_end   = sub.start + timedelta(seconds=(i+1) * dur_per_word)
            word_entries.append((w, w_start, w_end))
    return word_entries

########################################
# 5) Flatten lyrics → word list
########################################
def lyrics_to_word_list(lyrics_text):
    lines = lyrics_text.split('\n')
    word_list = []
    for line_idx, line in enumerate(lines):
        words = line.strip().split()
        for w in words:
            word_list.append((w, line_idx))
    return word_list, lines

########################################
# Normalization Helper
########################################
def normalize_word(word):
    w = word.lower()
    w = w.replace('-', ' ')
    w = ''.join(ch for ch in w if ch.isalnum() or ch.isspace())
    return w.strip()

########################################
# 6) Word-level alignment (DP)
########################################
def word_alignment(transcript_words, lyric_words):
    T = len(transcript_words)
    L = len(lyric_words)
    dp = [[(math.inf, None) for _ in range(T+1)] for _ in range(L+1)]
    dp[0][0] = (0, None)
    skip_penalty = 1

    def cost_function(lyric_word, transcript_word):
        lw = normalize_word(lyric_word)
        tw = normalize_word(transcript_word)
        return 0 if lw == tw else 1

    for i in range(L+1):
        for j in range(T+1):
            (curr_cost, _) = dp[i][j]
            if curr_cost == math.inf:
                continue

            # skip lyric word
            if i < L:
                scost = curr_cost + skip_penalty
                if scost < dp[i+1][j][0]:
                    dp[i+1][j] = (scost, (i, j))
            # skip transcript word
            if j < T:
                scost = curr_cost + skip_penalty
                if scost < dp[i][j+1][0]:
                    dp[i][j+1] = (scost, (i, j))

            # match
            if i < L and j < T:
                cst = cost_function(lyric_words[i][0], transcript_words[j][0])
                new_cost = curr_cost + cst
                if new_cost < dp[i+1][j+1][0]:
                    dp[i+1][j+1] = (new_cost, (i, j))

    alignment = [None]*L
    i, j = L, T
    while i > 0 or j > 0:
        cval, backptr = dp[i][j]
        if backptr is None:
            break
        pi, pj = backptr
        if pi == i-1 and pj == j-1:
            alignment[i-1] = j-1
        i, j = pi, pj
    return alignment

########################################
# 7) Build final SRT from alignment
########################################
def build_lyrics_srt_from_word_alignment(lyrics_text, alignment, transcript_words):
    lines = [l.strip() for l in lyrics_text.split('\n')]
    lyric_word_list, _ = lyrics_to_word_list(lyrics_text)
    line_to_word_indices = {}
    for i, (w, line_idx) in enumerate(lyric_word_list):
        if alignment[i] is not None:
            t_idx = alignment[i]
            if line_idx not in line_to_word_indices:
                line_to_word_indices[line_idx] = []
            line_to_word_indices[line_idx].append(t_idx)

    srt_entries = []
    idx_counter = 1
    for line_idx, line_text in enumerate(lines):
        matched_indices = line_to_word_indices.get(line_idx, [])
        if not matched_indices:
            continue
        start_time = min(transcript_words[mx][1] for mx in matched_indices)
        end_time   = max(transcript_words[mx][2] for mx in matched_indices)
        subtitle = srt.Subtitle(
            index=idx_counter,
            start=start_time,
            end=end_time,
            content=line_text
        )
        srt_entries.append(subtitle)
        idx_counter += 1
    srt_entries.sort(key=lambda x: x.start)
    return srt.compose(srt_entries)

########################################
# 8) Main alignment function
########################################
def lyrics_driven_srt(whisper_srt_str, correct_lyrics_text):
    transcript_word_list = srt_to_word_timestamps(whisper_srt_str)
    lyric_word_list, _ = lyrics_to_word_list(correct_lyrics_text)
    alignment = word_alignment(transcript_word_list, lyric_word_list)
    return build_lyrics_srt_from_word_alignment(correct_lyrics_text, alignment, transcript_word_list)

########################################
# Streamlit App
########################################

st.set_page_config(layout="wide")
st.title("TVG LyricsAI + Demucs")

# We'll store final data in st.session_state so it doesn't vanish on re-runs
if "vocals_bytes" not in st.session_state:
    st.session_state.vocals_bytes = None
if "srt_zip_bytes" not in st.session_state:
    st.session_state.srt_zip_bytes = None

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Use Demucs Vocal Isolation (preprocessing)?")

    # Let them pick 2 or 4 stems only if they are using Demucs
    if use_demucs:
        stems_choice = st.selectbox("Number of stems:", ["2 stems (vocals)", "4 stems (vocals/bass/drums/other)"])
    else:
        stems_choice = None

    st.markdown("[Need an API key?](https://platform.openai.com/account/api-keys)")

uploaded_audio = st.file_uploader(
    "Upload Audio File (mp3, wav, etc.)",
    type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"]
)
lyrics_input = st.text_area("Paste Lyrics (optional)")

if uploaded_audio:
    if not openai_api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()

    if st.button("Generate SRT"):
        # Reset stored data so each run is fresh
        st.session_state.vocals_bytes = None
        st.session_state.srt_zip_bytes = None

        # Step 1: Save audio to temp
        audio_path = save_temp_file(uploaded_audio, ".wav")
        audio_basename = os.path.splitext(uploaded_audio.name)[0]

        # Step 2: If Demucs is chosen, isolate vocals -> mp3
        # Decide 2 or 4 stems
        if use_demucs:
            two_stems = True
            if stems_choice and stems_choice.startswith("4"):
                two_stems = False

            with st.spinner("Running Demucs..."):
                vocals_mp3_path = run_demucs_vocals_mp3(
                    input_audio_path=audio_path,
                    two_stems=two_stems
                )

            # Save the vocals in session_state for later downloads
            with open(vocals_mp3_path, "rb") as vf:
                st.session_state.vocals_bytes = vf.read()

            processed_path = vocals_mp3_path
            mime_type = "audio/mpeg"
        else:
            processed_path = audio_path
            mime_type = "audio/wav"

        # Step 3: Transcribe with Whisper
        whisper_url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        data = {
            "model": "whisper-1",
            "response_format": "srt",
        }
        with st.spinner("Transcribing with Whisper..."):
            with open(processed_path, "rb") as f:
                files = {"file": (os.path.basename(processed_path), f, mime_type)}
                response = requests.post(whisper_url, headers=headers, files=files, data=data)
                response.raise_for_status()
            whisper_srt_str = response.text

        # Step 4: If user gave lyrics, align them
        if lyrics_input.strip():
            with st.spinner("Aligning to your pasted lyrics..."):
                final_srt_str = lyrics_driven_srt(whisper_srt_str, lyrics_input)
            # Create ZIP
            zip_path = create_zip(audio_basename, whisper_srt_str, final_srt_str)
        else:
            # Provide raw SRT in both slots
            zip_path = create_zip(audio_basename, whisper_srt_str, whisper_srt_str)

        # Store that zip data in session_state so the button persists
        with open(zip_path, "rb") as zf:
            st.session_state.srt_zip_bytes = zf.read()

        st.success("Done! Check below for your downloads.")

# Now show the buttons for downloads, if data is present:
if st.session_state.vocals_bytes:
    st.download_button(
        "Download Isolated Vocals (MP3)",
        data=st.session_state.vocals_bytes,
        file_name="vocals.mp3",
        mime="audio/mpeg"
    )

if st.session_state.srt_zip_bytes:
    st.download_button(
        "Download SRT Files (ZIP)",
        data=st.session_state.srt_zip_bytes,
        file_name="srt_files.zip",
        mime="application/zip"
    )
