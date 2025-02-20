import streamlit as st
from openai import OpenAI
import tempfile
import os
import srt
import zipfile
import math
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
# 2) Make ZIP with original + final
########################################
def create_zip(original_srt_str, aligned_srt_str):
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr("transcription.srt", original_srt_str)
        zipf.writestr("corrected_transcription.srt", aligned_srt_str)
    return zip_path

########################################
# 3) Convert Whisper SRT to word timestamps
########################################
def srt_to_word_timestamps(srt_text):
    """
    Parse the Whisper SRT into a flat list of (word, start_time, end_time).
    We'll naively distribute the subtitle's time range across its words.
    """
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
# 4) Flatten lyrics -> word list
########################################
def lyrics_to_word_list(lyrics_text):
    """
    Returns a list of (word, line_index) while preserving empty lines.
    Empty lines become lines with zero words, so they won't produce subtitles.
    """
    lines = lyrics_text.split('\n')  # keep empty lines
    word_list = []
    for line_idx, line in enumerate(lines):
        # strip each line just for splitting
        words = line.strip().split()
        for w in words:
            word_list.append((w, line_idx))
    return word_list, lines

########################################
# Normalization Helper
########################################
def normalize_word(word):
    """
    Lowercase, replace hyphens with spaces, remove non-alphanumeric except spaces,
    and strip leading/trailing whitespace.
    """
    w = word.lower()
    w = w.replace('-', ' ')
    w = ''.join(ch for ch in w if ch.isalnum() or ch.isspace())
    return w.strip()

########################################
# 5) Word-level alignment (DP)
########################################
def word_alignment(transcript_words, lyric_words):
    """
    Align transcript_words -> lyric_words with skipping allowed on both sides.
    transcript_words: list of (word, start_time, end_time)
    lyric_words: list of (word, line_idx)

    Return an array 'alignment' where alignment[i] = j means
    the i-th lyric word is matched with the j-th transcript word, or None if not matched.
    """
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

        # matched
        if pi == i-1 and pj == j-1:
            alignment[i-1] = j-1

        i, j = pi, pj

    return alignment

########################################
# 6) Build final SRT from alignment
########################################
def build_lyrics_srt_from_word_alignment(lyrics_text, alignment, transcript_words):
    lines = [l.strip() for l in lyrics_text.split('\n')]  # keep even empty lines (striped form)
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
            # no matched words => no subtitle
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
# 7) Main function to create lyrics-driven SRT
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
st.title("TVG LyricsAI")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Need an API key?](https://platform.openai.com/account/api-keys)"

uploaded_audio = st.file_uploader("Upload Audio File", 
                                  type=["flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"])
lyrics_input = st.text_area("Paste Correct Lyrics Here")

if uploaded_audio:
    if not openai_api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()

    if st.button("Generate SRT"):
        audio_path = save_temp_file(uploaded_audio, ".wav")

        client = OpenAI(api_key=openai_api_key)
        with open(audio_path, "rb") as audio_file:
            with st.spinner("Transcribing with Whisper..."):
                whisper_result_srt = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt"
                )
        whisper_srt_str = whisper_result_srt

        if lyrics_input.strip():
            with st.spinner("Aligning to your pasted lyrics..."):
                final_srt_str = lyrics_driven_srt(whisper_srt_str, lyrics_input)
            
            zip_path = create_zip(whisper_srt_str, final_srt_str)
            st.success("Done! Download your SRT files below.")
            st.download_button(
                "Download SRTs (ZIP)",
                data=open(zip_path, "rb"),
                file_name="srt_files.zip",
                mime="application/zip"
            )
        else:
            st.warning("No lyrics pasted! Showing only raw Whisper SRT.")
            zip_path = create_zip(whisper_srt_str, whisper_srt_str)
            st.download_button(
                "Download Raw Whisper SRT",
                data=open(zip_path, "rb"),
                file_name="transcribed.zip",
                mime="application/zip"
            )
