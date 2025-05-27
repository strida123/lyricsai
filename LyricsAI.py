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

# ──────────────────────────────────────────────────────────────────────────────
#  TVG LyricsAI + Demucs    —  now using GPT-4o-transcribe for speech-to-text
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st, requests, json, tempfile, os, srt, zipfile, math, subprocess
from datetime import timedelta
# ──────────────────────────  Utility: temporary files  ────────────────────────
def save_temp_file(uploaded_file, suffix):
    fn = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fn.write(uploaded_file.getvalue()); fn.close()
    return fn.name
# ──────────────────────────  Demucs isolation  ───────────────────────────────
def run_demucs_vocals_mp3(input_path, two_stems=True):
    out_dir = tempfile.mkdtemp()
    cmd = ["demucs", input_path, "-o", out_dir]
    if two_stems: cmd += ["--two-stems", "vocals"]
    subprocess.run(cmd, check=True)
    # locate vocals.wav produced by Demucs
    vocals_wav = next(
        (os.path.join(r, "vocals.wav") for r,_,f in os.walk(out_dir) if "vocals.wav" in f),
        None
    )
    if not vocals_wav:
        raise FileNotFoundError("Demucs finished but vocals.wav not found.")
    vocals_mp3 = vocals_wav.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, vocals_mp3], check=True)
    return vocals_mp3
# ──────────────────────────  ZIP helper  ──────────────────────────────────────
def create_zip(basename, original_srt, aligned_srt):
    zpath = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"{basename}_transcription.srt", original_srt)
        z.writestr(f"{basename}_corrected_transcription.srt", aligned_srt)
    return zpath
# ──────────────────────────  JSON → SRT  ──────────────────────────────────────
def json_to_srt_and_words(json_str):
    obj = json.loads(json_str)
    entries, word_list, idx = [], [], 1
    for seg in obj.get("segments", []):
        start = timedelta(seconds=seg["start"])
        end   = timedelta(seconds=seg["end"])
        txt   = seg["text"].strip()
        entries.append(srt.Subtitle(idx, start, end, txt)); idx += 1
        # word-level (if present)
        for w in seg.get("words", []):
            word_list.append((w["word"], timedelta(seconds=w["start"]),
                              timedelta(seconds=w["end"])))
    srt_str = srt.compose(entries)
    # if words array missing, fall back to per-segment coarse timing
    if not word_list:
        for e in entries:
            words = e.content.split()
            dur = (e.end - e.start).total_seconds() / max(len(words), 1)
            for i, w in enumerate(words):
                word_list.append((w,
                                  e.start + timedelta(seconds=i*dur),
                                  e.start + timedelta(seconds=(i+1)*dur)))
    return srt_str, word_list
# ──────────────────────────  Lyrics alignment helpers  ────────────────────────
def normalize_word(w): return ''.join(ch for ch in w.lower().replace('-', ' ')
                                      if ch.isalnum() or ch.isspace()).strip()
def lyrics_to_word_list(txt):
    lines, out = txt.splitlines(), []
    for li,line in enumerate(lines):
        for w in line.strip().split(): out.append((w, li))
    return out, lines
def word_alignment(trans_words, lyric_words):
    T,L = len(trans_words), len(lyric_words); skip = 1
    dp = [[(math.inf,None) for _ in range(T+1)] for _ in range(L+1)]
    dp[0][0]=(0,None)
    cost=lambda lw,tw:0 if normalize_word(lw)==normalize_word(tw) else 1
    for i in range(L+1):
        for j in range(T+1):
            c,_ = dp[i][j]
            if c==math.inf: continue
            if i<L and c+skip<dp[i+1][j][0]: dp[i+1][j]=(c+skip,(i,j))
            if j<T and c+skip<dp[i][j+1][0]: dp[i][j+1]=(c+skip,(i,j))
            if i<L and j<T:
                nc=c+cost(lyric_words[i][0], trans_words[j][0])
                if nc<dp[i+1][j+1][0]: dp[i+1][j+1]=(nc,(i,j))
    align=[None]*L; i,j=L,T
    while i or j:
        _,b=dp[i][j]; pi,pj=b or (0,0)
        if pi==i-1 and pj==j-1: align[i-1]=j-1
        i,j=pi,pj
    return align
def build_lyrics_srt(lyrics_text, align, trans_words):
    lines=[l.strip() for l in lyrics_text.splitlines()]
    lyric_words,_=lyrics_to_word_list(lyrics_text)
    line_map={}
    for i,(w,li) in enumerate(lyric_words):
        if align[i] is not None:
            line_map.setdefault(li,[]).append(align[i])
    subs=[]; idx=1
    for li,line in enumerate(lines):
        idxs=line_map.get(li,[])
        if not idxs: continue
        start=min(trans_words[k][1] for k in idxs)
        end  =max(trans_words[k][2] for k in idxs)
        subs.append(srt.Subtitle(idx,start,end,line)); idx+=1
    return srt.compose(sorted(subs,key=lambda s:s.start))
def align_lyrics(trans_words, lyrics_text):
    lyric_words,_=lyrics_to_word_list(lyrics_text)
    align=word_alignment(trans_words, lyric_words)
    return build_lyrics_srt(lyrics_text, align, trans_words)
# ──────────────────────────  Streamlit UI  ────────────────────────────────────
st.set_page_config(layout="wide"); st.title("TVG LyricsAI + Demucs (GPT-4o Edition)")
if "vocals_bytes" not in st.session_state: st.session_state.vocals_bytes=None
if "zip_bytes" not in st.session_state:    st.session_state.zip_bytes=None
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Isolate vocals with Demucs?")
    if use_demucs:
        stems_choice = st.selectbox("Demucs stems", ["2 stems (vocals)","4 stems"])
    else: stems_choice=None
    st.markdown("[Get an API key](https://platform.openai.com/account/api-keys)")
up_audio = st.file_uploader("Upload Audio", type=[
    "flac","m4a","mp3","mp4","mpeg","mpga","oga","ogg","wav","webm"])
lyrics_txt = st.text_area("Paste lyrics (optional)")
if up_audio and st.button("Generate SRT"):
    if not api_key: st.warning("Add your API key."); st.stop()
    # fresh state
    st.session_state.vocals_bytes=st.session_state.zip_bytes=None
    audio_path=save_temp_file(up_audio,".wav"); base=os.path.splitext(up_audio.name)[0]
    # Demucs if selected
    if use_demucs:
        with st.spinner("Running Demucs…"):
            vocals_mp3 = run_demucs_vocals_mp3(audio_path,
                          two_stems=not stems_choice.startswith("4"))
        with open(vocals_mp3,"rb") as f: st.session_state.vocals_bytes=f.read()
        proc_path, mime = vocals_mp3, "audio/mpeg"
    else:
        proc_path, mime = audio_path, "audio/wav"
    # ───────────  Transcribe with GPT-4o  ───────────
    model="gpt-4o-transcribe"
    headers={"Authorization":f"Bearer {api_key}"}
    data={
        "model":model,
        "response_format":"json",          # <-- critical fix
        "timestamp_granularities[]":"word" # ask for word stamps
    }
    with st.spinner("Transcribing with GPT-4o…"):
        with open(proc_path,"rb") as f:
            resp = requests.post("https://api.openai.com/v1/audio/transcriptions",
                                  headers=headers,
                                  files={"file":(os.path.basename(proc_path),f,mime)},
                                  data=data)
        # surface any error details
        try: resp.raise_for_status()
        except requests.HTTPError as e:
            st.error(f"OpenAI error: {resp.text or e}")
            st.stop()
    json_str = resp.text
    original_srt, trans_words = json_to_srt_and_words(json_str)
    # ───────────  optional lyrics alignment  ───────────
    if lyrics_txt.strip():
        with st.spinner("Aligning lyrics…"):
            final_srt = align_lyrics(trans_words, lyrics_txt)
    else:
        final_srt = original_srt
    zip_path = create_zip(base, original_srt, final_srt)
    with open(zip_path,"rb") as z: st.session_state.zip_bytes=z.read()
    st.success("All done!  Downloads below ↓")
# ──────────────────────────  Download buttons  ───────────────────────────────
if st.session_state.vocals_bytes:
    st.download_button("Download isolated vocals (MP3)",
        st.session_state.vocals_bytes,"vocals.mp3","audio/mpeg")
if st.session_state.zip_bytes:
    st.download_button("Download SRT files (ZIP)",
        st.session_state.zip_bytes,"captions.zip","application/zip")
