import streamlit as st
import requests
import srt
import zipfile
import math
import tempfile
import os
import subprocess
from datetime import timedelta

# ---- Spleeter stuff
from spleeter.separator import Separator

########################################
# 1) Save uploaded file
########################################
def save_temp_file(uploaded_file, suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name

########################################
# 2) Spleeter Preprocessing: Isolate Vocals -> Convert to MP3
########################################
def spleeter_vocals_to_mp3(input_audio_path):
    """
    Splits 'input_audio_path' into vocals & accompaniment.
    Writes 'vocals.wav' into a temp folder, then converts to 'vocals.mp3'.
    Returns the path to 'vocals.mp3'.
    """
    output_dir = tempfile.mkdtemp()
    # 1) Use Spleeter's Python API for 2 stems (vocals + accompaniment).
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_audio_path, output_dir)
    
    # 2) Spleeter will write 'vocals.wav' to a subfolder named after the base file
    basename = os.path.splitext(os.path.basename(input_audio_path))[0]
    vocals_wav = os.path.join(output_dir, basename, "vocals.wav")
    
    # 3) Convert WAV -> MP3 using ffmpeg
    vocals_mp3 = vocals_wav.replace(".wav", ".mp3")
    cmd_ffmpeg = ["ffmpeg", "-y", "-i", vocals_wav, vocals_mp3]
    subprocess.run(cmd_ffmpeg, check=True)
    
    return vocals_mp3

########################################
# 3) Make ZIP with original + final
########################################
def create_zip(audio_basename, original_srt_str, aligned_srt_str):
    """
    Creates a ZIP file containing:
      - {audio_basename}_transcription.srt
      - {audio_basename}_corrected_transcription.srt
    Returns the path to the .zip file.
    """
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
            # skip lyric
            if i < L:
                scost = curr_cost + skip_penalty
                if scost < dp[i+1][j][0]:
                    dp[i+1][j] = (scost, (i, j))
            # skip transcript
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
st.title("TVG LyricsAI with Spleeter")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Need an API key?](https://platform.openai.com/account/api-keys)")

uploaded_audio = st.file_uploader(
    "Upload Audio File (e.g. MP3)",
    type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"]
)
lyrics_input = st.text_area("Paste Lyrics (optional)")

if uploaded_audio:
    if not openai_api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()

    if st.button("Generate SRT"):
        # Step 1: Save uploaded audio to temp
        audio_path = save_temp_file(uploaded_audio, ".mp3")
        audio_basename = os.path.splitext(uploaded_audio.name)[0]

        # Step 2: Run Spleeter to isolate vocals, produce vocals.mp3
        with st.spinner("Isolating vocals via Spleeter..."):
            processed_mp3_path = spleeter_vocals_to_mp3(audio_path)

        # Step 3: Call Whisper via direct HTTP request
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
        }
        data = {
            "model": "whisper-1",
            "response_format": "srt",
        }
        with st.spinner("Transcribing vocals with Whisper..."):
            with open(processed_mp3_path, "rb") as f:
                files = {"file": (os.path.basename(processed_mp3_path), f, "audio/mpeg")}
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
            whisper_srt_str = response.text

        # Step 4: If user pasted lyrics, align them. Otherwise, just return raw SRT
        if lyrics_input.strip():
            with st.spinner("Aligning to your pasted lyrics..."):
                final_srt_str = lyrics_driven_srt(whisper_srt_str, lyrics_input)
            zip_path = create_zip(audio_basename, whisper_srt_str, final_srt_str)
            st.success("Done! Download your SRT files below.")
            st.download_button(
                "Download SRTs (ZIP)",
                data=open(zip_path, "rb"),
                file_name="srt_files.zip",
                mime="application/zip"
            )
        else:
            zip_path = create_zip(audio_basename, whisper_srt_str, whisper_srt_str)
            st.warning("No lyrics pasted! Providing only raw Whisper SRT.")
            st.download_button(
                "Download Raw Whisper SRT (ZIP)",
                data=open(zip_path, "rb"),
                file_name="transcribed.zip",
                mime="application/zip"
            )
