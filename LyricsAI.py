import streamlit as st, requests, json, tempfile, os, srt, zipfile, math, subprocess
from datetime import timedelta

# ─────────────────────────  Utility: temp files  ────────────────────────────

def save_temp_file(uploaded_file, suffix):
    """Save Streamlit UploadedFile to a temp file and return the path."""
    fn = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fn.write(uploaded_file.getvalue())
    fn.close()
    return fn.name

# ─────────────────────────  Demucs isolation  ───────────────────────────────

def run_demucs_vocals_mp3(input_path, two_stems=True):
    """Run Demucs on *input_path* and return vocals.mp3 path."""
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
    if not vocals_wav:
        raise FileNotFoundError("vocals.wav not found after Demucs run.")

    vocals_mp3 = vocals_wav.replace(".wav", ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, vocals_mp3], check=True)
    return vocals_mp3

# ─────────────────────────  ZIP helper  ──────────────────────────────────────

def create_zip(basename, original_srt, aligned_srt):
    """Create a ZIP containing *original_srt* and *aligned_srt*; return path."""
    zpath = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(f"{basename}_transcription.srt", original_srt)
        z.writestr(f"{basename}_corrected_transcription.srt", aligned_srt)
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
            words.append((w["word"], timedelta(seconds=w["start"]), timedelta(seconds=w["end"])))

    # Fallback: synthesize word timings if API didn't return "words"
    if not words:
        for s in subs:
            tokens = s.content.split()
            dur = (s.end - s.start).total_seconds() / max(len(tokens), 1)
            for i, t in enumerate(tokens):
                words.append((t, s.start + timedelta(seconds=i*dur), s.start + timedelta(seconds=(i+1)*dur)))

    return srt.compose(subs), words

# ─────────────────────────  Lyrics alignment helpers  ───────────────────────

def normalize_word(word):
    return "".join(ch for ch in word.lower().replace("-", " ") if ch.isalnum() or ch.isspace()).strip()


def lyrics_to_word_list(text):
    lines = text.splitlines()
    out = []
    for li, line in enumerate(lines):
        for w in line.strip().split():
            out.append((w, li))
    return out, lines


def word_alignment(trans_words, lyric_words):
    T, L, skip = len(trans_words), len(lyric_words), 1
    dp = [[(math.inf, None) for _ in range(T+1)] for _ in range(L+1)]
    dp[0][0] = (0, None)

    for i in range(L+1):
        for j in range(T+1):
            c, _ = dp[i][j]
            if c == math.inf:
                continue
            if i < L and c + skip < dp[i+1][j][0]:
                dp[i+1][j] = (c + skip, (i, j))
            if j < T and c + skip < dp[i][j+1][0]:
                dp[i][j+1] = (c + skip, (i, j))
            if i < L and j < T:
                nc = c + (0 if normalize_word(lyric_words[i][0]) == normalize_word(trans_words[j][0]) else 1)
                if nc < dp[i+1][j+1][0]:
                    dp[i+1][j+1] = (nc, (i, j))

    align = [None] * L
    i, j = L, T
    while i or j:
        _, back = dp[i][j]
        if back is None:
            break
        pi, pj = back
        if pi == i - 1 and pj == j - 1:
            align[i-1] = j-1
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
st.title("TVG LyricsAI + Demucs (GPT-4o Edition)")

if "vocals_bytes" not in st.session_state:
    st.session_state.vocals_bytes = None
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    use_demucs = st.checkbox("Isolate vocals with Demucs?")
    stems_choice = None
    if use_demucs:
        stems_choice = st.selectbox("Demucs stems", ["2 stems (vocals)", "4 stems"])
    st.markdown("[Get an API key](https://platform.openai.com/account/api-keys)")

up_audio = st.file_uploader("Upload Audio", type=[
    "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"
])
lyrics_txt = st.text_area("Paste lyrics (optional)")

if up_audio and st.button("Generate SRT"):
    if not api_key:
        st.warning("Add your API key.")
        st.stop()

    st.session_state.vocals_bytes = st.session_state.zip_bytes = None
    file_ext = os.path.splitext(up_audio.name)[1] or ".wav"
    audio_path = save_temp_file(up_audio, file_ext)
    base = os.path.splitext(up_audio.name)[0]

    # Demucs (optional)
    if use_demucs:
        two_stems = not (stems_choice and stems_choice.startswith("4"))
        with st.spinner("Running Demucs …"):
            vocals_mp3 = run_demucs_vocals_mp3(audio_path, two_stems=two_stems)
        with open(vocals_mp3, "rb") as f:
            st.session_state.vocals_bytes = f.read()
        proc_path, mime = vocals_mp3, "audio/mpeg"
    else:
        proc_path, mime = audio_path, (up_audio.type or "audio/wav")

    # GPT-4o transcription
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
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            st.error(f"OpenAI error: {resp.text or e}")
            st.stop()

    original_srt, trans_words = json_to_srt_and_words(resp.text)

    # Lyrics alignment (optional)
    if lyrics_txt.strip():
        with st.spinner("Aligning lyrics …"):
            final_srt = align_lyrics(trans_words, lyrics_txt)
    else:
        final_srt = original_srt

    # Package ZIP
    zip_path = create_zip(base, original_srt, final_srt)
    with open(zip_path, "rb") as z:
        st.session_state.zip_bytes = z.read()

    st.success("All done! Downloads below ↓")

# ─────────────────────────  Download buttons  ───────────────────────────────

if st.session_state.vocals_bytes:
    st.download_button(
        "Download isolated vocals (MP3)",
        st.session_state.vocals_bytes,
        "vocals.mp3",
        "audio/mpeg",
    )

if st.session_state.zip_bytes:
    st
