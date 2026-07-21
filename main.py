"""
DeepLearn TTS — timing-synced voiceover service (personal use)

Pipeline (all in this one service):
  video  ->  extract audio (ffmpeg)
         ->  transcribe with word/segment timestamps (Groq Whisper)
         ->  translate each segment to Burmese (Gemini 2.5 Flash)
         ->  synthesize each segment (Azure Neural TTS, SSML rate control)
         ->  HYBRID +-20% timeline assembler (pydub) -> place each clip at its
             original start time, keep silence where the source is silent
         ->  return  output.mp3 + output.srt  (as a zip)

No auth, no database. Keys live in Render env vars.

Required env vars:
  AZURE_SPEECH_KEY      your Azure Speech key
  AZURE_SPEECH_REGION   e.g. southeastasia
  GROQ_API_KEY          for Whisper transcription
  GEMINI_API_KEY        for translation

Requires ffmpeg on the host (see README).
"""

import io
import os
import re
import json
import math
import time
import shutil
import zipfile
import tempfile
import subprocess
from typing import List, Dict

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# ---------------------------------------------------------------- config

AZURE_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "southeastasia")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# Which transcriber to use: "assemblyai" (default, no daily audio limit) or "groq".
TRANSCRIBER = os.environ.get("TRANSCRIBER", "assemblyai").lower()
# Recap style: "1" = video-aware (Gemini watches the video); "0" = FIRST STYLE
# (transcript-based: AssemblyAI transcribe -> Gemini recap, Gemini never sees the video).
# Default is the FIRST STYLE (transcript) — Thet's preferred style.
RECAP_VIDEO_AWARE = os.environ.get("RECAP_VIDEO_AWARE", "0") == "1"

GEMINI_MODEL = "gemini-2.5-flash"          # not lite, not 1.5 (deprecated)
GROQ_MODEL = "whisper-large-v3"
DEFAULT_VOICE = "my-MM-NilarNeural"        # or my-MM-ThihaNeural
# Default Azure voice per output language. The SSML locale (xml:lang) is derived
# from the voice name, so any Azure voice works — just pass its name.
DEFAULT_VOICES = {
    "Burmese": "my-MM-NilarNeural",        # female · male: my-MM-ThihaNeural
    "English": "en-US-GuyNeural",          # male   · female: en-US-JennyNeural
    "Chinese": "zh-CN-YunxiNeural",        # male   · female: zh-CN-XiaoxiaoNeural
}

# Hybrid timing knob: how much we are allowed to speed a clip up to fit its slot.
# Full/faithful translations are often longer than the source, so allow a
# listenable ceiling (+40%). Beyond this we don't push faster (it gets unclear) —
# the extra spills into the following silence gap instead.
MAX_SPEEDUP = 0.40                         # +40% (dub mode)
# Recap narration should stay natural — barely nudge it, and rely on
# duration-calibrated script length instead of speed to keep pace with the video.
RECAP_SPEEDUP = 0.15                        # +15% (recap mode)
# If a clip is only slightly longer than its slot, don't bother re-synthesizing.
SLACK = 0.05                               # 5%

app = FastAPI(title="DeepLearn timing-synced TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                   # personal use; tighten if you host the UI
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers

def _run(cmd: List[str]):
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:500])
    return p


def extract_audio(video_path: str, out_path: str):
    """Extract a small mono 16kHz mp3 so it stays under Groq's 25MB limit."""
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        out_path,
    ])


def media_duration_ms(path: str) -> int:
    """Total media length via ffprobe — used so trailing silence is preserved."""
    p = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    return int(float(p.stdout.decode().strip()) * 1000)


# ---------------------------------------------------------------- transcription

def _groq_keys() -> List[str]:
    """Collect all configured Groq keys so we can rotate when one hits its
    daily audio limit. Reads GROQ_API_KEY plus GROQ_KEY_1..GROQ_KEY_5."""
    keys = []
    for name in ["GROQ_API_KEY", "GROQ_KEY_1", "GROQ_KEY_2",
                 "GROQ_KEY_3", "GROQ_KEY_4", "GROQ_KEY_5"]:
        v = os.environ.get(name, "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def transcribe_groq(audio_path: str) -> List[Dict]:
    """Groq Whisper -> [{start_ms, end_ms, text}]. Rotates across keys: if one
    key is rate-limited (429), tries the next before giving up."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    fname = os.path.basename(audio_path)

    keys = _groq_keys()
    if not keys:
        raise HTTPException(500, "No Groq API key configured.")

    last_err = ""
    for key in keys:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (fname, audio_bytes, "audio/mpeg")},
            data={
                "model": GROQ_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=600,
        )
        if r.status_code == 200:
            segs = r.json().get("segments", [])
            out = []
            for s in segs:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                out.append({
                    "start_ms": int(s["start"] * 1000),
                    "end_ms": int(s["end"] * 1000),
                    "text": text,
                })
            return out
        last_err = r.text[:300]
        if r.status_code == 429:
            continue          # this key is rate-limited today; try the next one
        break                 # other errors won't be fixed by another key
    raise HTTPException(502, f"Groq transcription failed (all keys tried): {last_err}")


def transcribe_assemblyai(audio_path: str) -> List[Dict]:
    """AssemblyAI -> [{start_ms, end_ms, text}] at sentence level. No daily audio
    limit like Groq. Uploads the audio, starts a transcript, polls until done, then
    pulls sentence-level timestamps (already in milliseconds)."""
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(500, "No AssemblyAI API key configured (ASSEMBLYAI_API_KEY).")
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    base = "https://api.assemblyai.com/v2"

    # 1) upload the audio file
    with open(audio_path, "rb") as f:
        up = requests.post(f"{base}/upload", headers=headers, data=f, timeout=600)
    if up.status_code != 200:
        raise HTTPException(502, f"AssemblyAI upload failed: {up.text[:300]}")
    audio_url = up.json()["upload_url"]

    # 2) start the transcript
    tr = requests.post(f"{base}/transcript", headers=headers,
                       json={"audio_url": audio_url}, timeout=60)
    if tr.status_code != 200:
        raise HTTPException(502, f"AssemblyAI transcript request failed: {tr.text[:300]}")
    tid = tr.json()["id"]

    # 3) poll until completed (up to ~10 min)
    for _ in range(300):
        pr = requests.get(f"{base}/transcript/{tid}", headers=headers, timeout=60)
        j = pr.json()
        status = j.get("status")
        if status == "completed":
            break
        if status == "error":
            raise HTTPException(502, f"AssemblyAI error: {j.get('error')}")
        time.sleep(2)
    else:
        raise HTTPException(504, "AssemblyAI transcription timed out.")

    # 4) sentence-level timestamps (already in ms)
    out = []
    sr = requests.get(f"{base}/transcript/{tid}/sentences", headers=headers, timeout=60)
    if sr.status_code == 200 and sr.json().get("sentences"):
        for s in sr.json()["sentences"]:
            text = (s.get("text") or "").strip()
            if text:
                out.append({"start_ms": int(s["start"]), "end_ms": int(s["end"]), "text": text})
    else:
        # fallback: group words into ~pseudo-sentences if the sentences call is empty
        buf, bstart, blast = [], None, None
        for w in (j.get("words") or []):
            if bstart is None:
                bstart = w["start"]
            buf.append(w["text"])
            blast = w["end"]
            if w["text"].endswith((".", "?", "!", "。", "？", "！")) or len(buf) >= 18:
                out.append({"start_ms": int(bstart), "end_ms": int(blast),
                            "text": " ".join(buf).strip()})
                buf, bstart, blast = [], None, None
        if buf:
            out.append({"start_ms": int(bstart), "end_ms": int(blast),
                        "text": " ".join(buf).strip()})
    return out


def transcribe(audio_path: str) -> List[Dict]:
    """Dispatch to the selected transcriber (default AssemblyAI)."""
    if TRANSCRIBER == "groq":
        return transcribe_groq(audio_path)
    return transcribe_assemblyai(audio_path)


# ------------------------------------------------- translation / recap script

def _gemini_json(prompt: str):
    """One Gemini call that must return a JSON array of {i, t}. Enforces a schema
    and salvages slightly-malformed replies instead of crashing."""
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {"i": {"type": "INTEGER"}, "t": {"type": "STRING"}},
                        "required": ["i", "t"],
                    },
                },
                "maxOutputTokens": 8192,
            },
        },
        timeout=300,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini failed: {r.text[:300]}")
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    # 1) normal parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 2) trim to the outermost [...] and retry
    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(raw[a:b + 1])
        except Exception:
            pass
    # 3) last resort: parse each {...} object individually, skip broken ones
    items = []
    for mo in re.finditer(r'\{[^{}]*?"i"\s*:\s*\d+[^{}]*?\}', raw, flags=re.DOTALL):
        try:
            items.append(json.loads(mo.group(0)))
        except Exception:
            continue
    if items:
        return items
    raise HTTPException(502, "Gemini returned unparseable JSON.")


def translate_segments(segments: List[Dict], target: str = "Burmese") -> None:
    """DUB mode: literal, concise, duration-aware translation of each line."""
    if not segments:
        return
    numbered = [
        {"i": i, "sec": round((s["end_ms"] - s["start_ms"]) / 1000, 1), "text": s["text"]}
        for i, s in enumerate(segments)
    ]
    prompt = (
        f"Translate each item's 'text' into natural, FAITHFUL spoken {target} for a "
        "voiceover — keep the FULL meaning, do not omit or summarize content. It is OK if a "
        "translation runs a little long; the voice will be sped up to fit its 'sec' seconds. "
        "Just write complete, natural {t}. Return ONLY a JSON array of objects like "
        "{{\"i\": 0, \"t\": \"...\"}}, same length and order, no markdown, no commentary.\n\n"
        .format(t=target)
        + json.dumps(numbered, ensure_ascii=False)
    )
    items = _gemini_json(prompt)
    by_i = {int(it["i"]): it.get("t", "") for it in items}
    for i, s in enumerate(segments):
        s["speak"] = by_i.get(i, s["text"]).strip()


def build_windows(segments: List[Dict], target_ms: int = 12000,
                  max_gap_ms: int = 1500) -> List[Dict]:
    """Merge consecutive transcript segments into ~target_ms scene windows.
    A long silence gap (> max_gap_ms) forces a new window (scene boundary)."""
    windows: List[Dict] = []
    cur = None
    for s in segments:
        if cur is None:
            cur = {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["text"]}
            continue
        gap = s["start_ms"] - cur["end_ms"]
        span = s["end_ms"] - cur["start_ms"]
        if gap > max_gap_ms or span > target_ms:
            windows.append(cur)
            cur = {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["text"]}
        else:
            cur["end_ms"] = s["end_ms"]
            cur["text"] += " " + s["text"]
    if cur:
        windows.append(cur)
    return windows


def rescript_recap(windows: List[Dict], target: str = "Burmese") -> None:
    """RECAP mode: rewrite the timestamped transcript as a THIRD-PERSON NARRATOR
    recap — one timed beat per scene window, flowing as one continuous story with
    an opening hook. Sets window["speak"]."""
    if not windows:
        return
    numbered = [
        {"i": i, "sec": round((w["end_ms"] - w["start_ms"]) / 1000, 1), "scene": w["text"]}
        for i, w in enumerate(windows)
    ]
    prompt = (
        f"You are scripting a MOVIE RECAP narration in {target}. Below is a timestamped "
        "transcript split into scene windows — each has an index 'i', a duration 'sec', and "
        "the original 'scene' text.\n"
        "Rewrite it as an engaging THIRD-PERSON NARRATOR recap — NOT a literal translation of "
        "the dialogue. Tell the story: what happens, who does what, the stakes and turns. "
        "Item 0 must open with a short punchy HOOK that makes viewers want to keep watching.\n"
        "LENGTH CALIBRATION (most important): each beat must be speakable at a NATURAL, unhurried "
        "narrator pace within its 'sec' seconds — roughly 3 to 4 syllables per second, so "
        "about (sec x 3) syllables. Aim to FILL the time with a steady continuous story; do not "
        "leave it too short. But do NOT write more than fits; "
        "if a window is short, use only a few words. The beats are played one after another as a "
        "continuous recap, so keep a steady flow.\n"
        "Rules:\n"
        "- Exactly one narration beat per window, SAME index and order.\n"
        f"- Natural spoken {target}, storytelling tone. No scene numbers, no stage directions.\n"
        "Return ONLY a JSON array of objects like {\"i\": 0, \"t\": \"...\"}, same length and "
        "order, no markdown, no commentary.\n\n"
        + json.dumps(numbered, ensure_ascii=False)
    )
    items = _gemini_json(prompt)
    by_i = {int(it["i"]): it.get("t", "") for it in items}
    for i, w in enumerate(windows):
        w["speak"] = by_i.get(i, w["text"]).strip()


# ------------------------------------------------- video-aware recap (Gemini)

def _salvage_json(raw: str):
    """Parse a JSON array, salvaging slightly-malformed replies."""
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(raw[a:b + 1])
        except Exception:
            pass
    items = []
    for mo in re.finditer(r'\{[^{}]*?\}', raw, flags=re.DOTALL):
        try:
            items.append(json.loads(mo.group(0)))
        except Exception:
            continue
    if items:
        return items
    raise HTTPException(502, "Gemini returned unparseable JSON.")


def _gemini_upload_video(path: str, mime: str = "video/mp4"):
    """Upload a video to the Gemini File API (resumable) and wait until ACTIVE."""
    num = os.path.getsize(path)
    start = requests.post(
        f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={GEMINI_API_KEY}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": "clip"}}, timeout=60,
    )
    upload_url = start.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise HTTPException(502, f"Gemini upload start failed: {start.text[:200]}")
    with open(path, "rb") as f:
        up = requests.post(upload_url, headers={
            "Content-Length": str(num),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }, data=f, timeout=1800)
    if up.status_code != 200:
        raise HTTPException(502, f"Gemini upload failed: {up.text[:200]}")
    info = up.json()["file"]
    name, uri, mime2 = info["name"], info.get("uri"), info.get("mimeType", mime)
    for _ in range(150):                                   # wait for processing (~5 min)
        g = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/{name}?key={GEMINI_API_KEY}",
            timeout=30).json()
        if g.get("state") == "ACTIVE":
            return g.get("uri", uri), g.get("mimeType", mime2)
        if g.get("state") == "FAILED":
            raise HTTPException(502, "Gemini video processing failed.")
        time.sleep(2)
    raise HTTPException(504, "Gemini video processing timed out.")


def gemini_video_recap(video_path: str, total_ms: int, target: str = "Burmese") -> List[Dict]:
    """VIDEO-AWARE recap: Gemini watches the video (visuals + audio) and returns
    timed beats synced to what happens on screen. Returns [{start_ms,end_ms,text,speak}]."""
    uri, mime = _gemini_upload_video(video_path)
    secs = round(total_ms / 1000, 1)
    prompt = (
        f"Watch this video fully — the visuals on screen AND the audio. Write an engaging "
        f"MOVIE RECAP narration in {target}, in a THIRD-PERSON NARRATOR voice (NOT a literal "
        "dub of the dialogue). Base every beat on what actually happens ON SCREEN at that moment.\n"
        "Requirements:\n"
        "- Break the recap into timed beats that follow the video's REAL timeline. Each beat has "
        "s = start time in seconds and e = end time in seconds, matching WHEN that action happens "
        f"on screen, and t = the {target} narration for that beat.\n"
        "- The FIRST beat (starting at 0) must be a punchy HOOK matching the opening action.\n"
        f"- Cover the WHOLE video from 0 to about {secs} seconds, no long gaps, in time order.\n"
        "- Each beat's narration must be speakable within (e - s) seconds at a natural narrator "
        "pace (~3-4 syllables/sec) — keep beats tight and synced to the action; mention key "
        "visible details (who does what, important objects/moments).\n"
        f"- Natural spoken {target}, storytelling tone. No stage directions, no emojis.\n"
        'Return ONLY a JSON array of {"s": <sec>, "e": <sec>, "t": "<narration>"} in time order.'
    )
    body = {
        "contents": [{"parts": [
            {"fileData": {"mimeType": mime, "fileUri": uri}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "temperature": 0.5,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"s": {"type": "NUMBER"}, "e": {"type": "NUMBER"},
                                   "t": {"type": "STRING"}},
                    "required": ["s", "e", "t"],
                },
            },
            "maxOutputTokens": 8192,
        },
    }
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body, timeout=600,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini video recap failed: {r.text[:300]}")
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    items = _salvage_json(raw)
    beats = []
    for it in items:
        try:
            s = int(float(it["s"]) * 1000)
            e = int(float(it["e"]) * 1000)
            t = (it.get("t") or "").strip()
        except Exception:
            continue
        if t and e > s:
            beats.append({"start_ms": s, "end_ms": e, "text": t, "speak": t})
    beats.sort(key=lambda b: b["start_ms"])
    return beats


def make_recap_beats(video_path: str, total_ms: int, target: str) -> List[Dict]:
    """Video-aware recap (Gemini watches the video). If RECAP_VIDEO_AWARE is off, or
    if video-aware fails, use the FIRST STYLE: transcript-based recap (Gemini never
    sees the video)."""
    if RECAP_VIDEO_AWARE:
        try:
            beats = gemini_video_recap(video_path, total_ms, target)
            if beats:
                print(f"[recap] VIDEO-AWARE ok — {len(beats)} beats from Gemini watching the video")
                return beats
        except Exception as e:
            print(f"[recap] video-aware FAILED, falling back to transcript: {str(e)[:200]}")
    # first style: transcript-based recap (Gemini does NOT watch the video)
    print("[recap] TRANSCRIPT style (Gemini does not watch the video)")
    audio_path = video_path + ".audio.mp3"
    extract_audio(video_path, audio_path)
    segs = transcribe(audio_path)
    if not segs:
        raise HTTPException(422, "No speech/visuals usable for a recap.")
    windows = build_windows(segs)
    rescript_recap(windows, target)
    return windows

def _locale_from_voice(voice: str) -> str:
    """en-US-GuyNeural -> en-US ; my-MM-NilarNeural -> my-MM (SSML xml:lang)."""
    parts = voice.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else "en-US"


def _ssml(text: str, voice: str, rate_pct: int) -> str:
    rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    # strip characters that are invalid in XML (control chars break Azure's SSML)
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 0x20)
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    lang = _locale_from_voice(voice)
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xml:lang='{lang}'>"
        f"<voice name='{voice}'><prosody rate='{rate}'>{safe}</prosody></voice>"
        "</speak>"
    )


def azure_tts(text: str, voice: str, rate_pct: int = 0) -> AudioSegment:
    """One Azure REST call -> AudioSegment (24kHz mono wav)."""
    endpoint = f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    r = requests.post(
        endpoint,
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            "User-Agent": "deeplearn-tts",
        },
        data=_ssml(text, voice, rate_pct).encode("utf-8"),
        timeout=120,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Azure TTS failed ({r.status_code}): {r.text[:200]}")
    if not r.content or len(r.content) < 100:            # 200 but empty/non-audio body
        print(f"[azure] empty audio for text: {text[:60]!r} — skipped")
        return None                                       # let caller substitute silence
    try:
        return AudioSegment.from_file(io.BytesIO(r.content), format="wav")
    except Exception:
        return None


def synth_segment_hybrid(text: str, slot_ms: int, voice: str,
                         max_speedup: float = MAX_SPEEDUP) -> AudioSegment:
    """
    HYBRID strategy (full translation + speed-up to fit):
      1. synthesize at natural rate, measure it
      2. if it fits the slot (within +5%): keep as-is (natural voice)
      3. if it's too long: re-synthesize ONCE with Azure rate, capped at MAX_SPEEDUP
         (Azure changes tempo without wrecking pitch — better than ffmpeg atempo)
      4. anything still over the cap spills into the following silence gap
    Short clips are never stretched to fill — we just leave trailing silence.
    """
    if not text or not text.strip():                 # empty beat -> short silence, skip Azure
        return AudioSegment.silent(duration=min(max(slot_ms, 200), 800), frame_rate=24000)
    clip = azure_tts(text, voice, 0)
    if clip is None:                                 # Azure gave no audio for this beat
        return AudioSegment.silent(duration=min(max(slot_ms, 200), 1200), frame_rate=24000)
    if slot_ms <= 0 or len(clip) <= slot_ms * (1 + SLACK):
        return clip
    needed = len(clip) / slot_ms - 1.0          # e.g. 0.35 => 35% too long
    rate_pct = int(round(min(needed, max_speedup) * 100))
    if rate_pct <= 0:
        return clip
    faster = azure_tts(text, voice, rate_pct)
    return faster if faster is not None else clip


# ---------------------------------------------------------------- assembler

def build_timeline(segments: List[Dict], total_ms: int, voice: str) -> AudioSegment:
    """
    DUB assembler. Each clip is anchored near its start_ms, but a cursor GUARANTEES
    no two clips ever play at once (fixes the "double voice" overlap): a clip starts
    at max(start_ms, end-of-previous-clip). Clips are fit (hybrid) to the room up to
    the next line so drift stays small.
    """
    base = AudioSegment.silent(duration=total_ms, frame_rate=24000)
    n = len(segments)
    cursor = 0
    for i, s in enumerate(segments):
        next_start = segments[i + 1]["start_ms"] if i + 1 < n else total_ms
        room = max(s["end_ms"] - s["start_ms"], next_start - s["start_ms"])
        clip = synth_segment_hybrid(s["speak"], room, voice)
        place_at = max(s["start_ms"], cursor)          # never overlap the previous clip
        s["play_start"] = place_at
        s["play_end"] = place_at + len(clip)
        s["tts_dur_ms"] = len(clip)
        s["room_ms"] = room
        # base may be shorter than place_at+clip; extend if needed
        need = place_at + len(clip)
        if need > len(base):
            base += AudioSegment.silent(duration=need - len(base), frame_rate=24000)
        base = base.overlay(clip, position=place_at)
        cursor = place_at + len(clip)
    return base


def build_timeline_recap(windows: List[Dict], total_ms: int, voice: str,
                         anchor: bool = False) -> AudioSegment:
    """
    RECAP assembler. anchor=False: beats play back-to-back continuously (no stops).
    anchor=True (video-aware): soft-anchor each beat to its real start time — wait in
    silence if we're ahead of the scene, but NEVER overlap the previous beat — so the
    narration stays synced to what's on screen. Records actual play times for the SRT.
    """
    out = AudioSegment.silent(duration=0, frame_rate=24000)
    cursor = 0
    n = len(windows)
    for i, w in enumerate(windows):
        if anchor and w["start_ms"] > cursor:              # ahead of the scene -> wait
            out += AudioSegment.silent(duration=w["start_ms"] - cursor, frame_rate=24000)
            cursor = w["start_ms"]
        if anchor:
            nxt = windows[i + 1]["start_ms"] if i + 1 < n else total_ms
            room = max(1500, nxt - cursor)                 # room up to the next scene
        else:
            room = max(1500, w["end_ms"] - w["start_ms"])
        clip = synth_segment_hybrid(w["speak"], room, voice, max_speedup=RECAP_SPEEDUP)
        w["play_start"] = cursor
        w["play_end"] = cursor + len(clip)
        w["tts_dur_ms"] = len(clip)
        w["room_ms"] = room
        out += clip
        cursor += len(clip)
    if total_ms > cursor:
        out += AudioSegment.silent(duration=total_ms - cursor, frame_rate=24000)
    return out


def build_srt(segments: List[Dict], speed: float = 1.0) -> str:
    """SRT that matches the FINAL audio. Cue times use actual play times when known,
    and are divided by `speed` because the whole track was sped up by that factor."""
    def ts(ms):
        ms = int(ms / max(speed, 0.01))                # match sped-up audio
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    lines = []
    for i, s in enumerate(segments, 1):
        start = s.get("play_start", s["start_ms"])     # actual audio time if known
        end = s.get("play_end", s["end_ms"])
        lines.append(str(i))
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(s.get("speak", s.get("text", "")))
        lines.append("")
    return "\n".join(lines)


def apply_speed(audio: AudioSegment, factor: float) -> AudioSegment:
    """Speed up (or down) the whole track WITHOUT changing pitch (ffmpeg atempo).
    Matches the usual recap workflow of speeding the video up ~1.1-1.4x — you then
    speed the VIDEO by the same factor when editing so they stay in sync."""
    if abs(factor - 1.0) < 0.01:
        return audio
    factor = max(0.5, min(2.0, factor))          # atempo single-filter range
    d = tempfile.mkdtemp(prefix="spd_")
    try:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        audio.export(src, format="wav")
        _run(["ffmpeg", "-y", "-i", src, "-filter:a", f"atempo={factor}", "-vn", dst])
        return AudioSegment.from_file(dst, format="wav")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def analyze(beats: List[Dict], total_ms: int, mode: str) -> Dict:
    """Compute how well the voiceover fit the video — the numbers behind good vs bad."""
    speech_ms = sum(b["end_ms"] - b["start_ms"] for b in beats)
    fills, overflow = [], 0
    for b in beats:
        room, dur = b.get("room_ms"), b.get("tts_dur_ms")
        if room and dur:
            f = dur / room
            fills.append(f)
            if f > 1.05:
                overflow += 1
    return {
        "mode": mode,
        "video_sec": round(total_ms / 1000, 1),
        "speech_density": round(speech_ms / max(total_ms, 1), 2),   # 1.0 = wall-to-wall talk
        "beats": len(beats),
        "avg_fill": round(sum(fills) / len(fills), 2) if fills else None,  # >1 = too much to say
        "max_fill": round(max(fills), 2) if fills else None,
        "overflow_beats": overflow,                                 # beats that needed to overflow
    }


def analysis_report(metrics: Dict) -> str:
    """Ask Gemini for a short human verdict + recommended settings for this video."""
    prompt = (
        "You are a QA assistant for a Burmese movie-recap voiceover tool. Given these metrics "
        "for ONE video's generated voiceover, write a SHORT analysis in Burmese (3-5 sentences): "
        "is the result likely good or too rushed, and why (reference speech_density / avg_fill / "
        "overflow_beats)? Then recommend the best mode ('recap' or 'dub') and a global output "
        "speed between 1.0 and 1.4 for THIS video. Be concrete.\n\n"
        "Metrics: " + json.dumps(metrics)
    )
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.4}},
            timeout=120,
        )
        verdict = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:                                          # analysis is best-effort
        verdict = f"(analysis unavailable: {e})"
    return "METRICS\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n\nVERDICT\n" + verdict


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg"))}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    voice: str = Form(""),              # blank -> auto-pick by target_lang
    target_lang: str = Form("Burmese"),
    mode: str = Form("recap"),          # "recap" (narrator) or "dub" (literal)
    speed: float = Form(1.0),           # global output speed (1.0-1.4 typical)
):
    """Full pipeline. Returns a zip with output.mp3 + output.srt + analysis.txt."""
    if not voice:
        voice = DEFAULT_VOICES.get(target_lang, DEFAULT_VOICE)
    workdir = tempfile.mkdtemp(prefix="dltts_")
    try:
        video_path = os.path.join(workdir, file.filename or "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        audio_path = os.path.join(workdir, "audio.mp3")
        total_ms = media_duration_ms(video_path)

        if mode == "dub":
            extract_audio(video_path, audio_path)
            beats = transcribe(audio_path)
            if not beats:
                raise HTTPException(422, "No speech detected in the video.")
            translate_segments(beats, target_lang)      # sets beat["speak"]
            timeline = build_timeline(beats, total_ms, voice)          # anchored
        else:                                            # recap
            beats = make_recap_beats(video_path, total_ms, target_lang)
            timeline = build_timeline_recap(beats, total_ms, voice)  # continuous, no stops
            fit = len(timeline) / max(1, total_ms)       # auto-fit to the video length
            timeline = apply_speed(timeline, max(1.0, min(fit, 1.8)))

        metrics = analyze(beats, total_ms, mode)         # before global speed
        timeline = apply_speed(timeline, speed)          # global recap-style speed-up
        metrics["global_speed"] = speed

        mp3_buf = io.BytesIO()
        timeline.export(mp3_buf, format="mp3", bitrate="128k")
        srt_text = build_srt(beats, speed)
        report = analysis_report(metrics)
        script = {
            "total_ms": total_ms, "mode": mode, "voice": voice,
            "target_lang": target_lang, "speed": speed,
            "beats": [{"i": i, "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                       "text": b.get("speak", "")} for i, b in enumerate(beats)],
        }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("output.mp3", mp3_buf.getvalue())
            z.writestr("output.srt", srt_text)
            z.writestr("analysis.txt", report)
            z.writestr("script.json", json.dumps(script, ensure_ascii=False, indent=2))
        zip_buf.seek(0)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="voiceover.zip"'},
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/script")
async def script(
    file: UploadFile = File(...),
    target_lang: str = Form("Burmese"),
    mode: str = Form("recap"),
):
    """Step 1: transcribe + write the script only (NO voice). Returns editable JSON.
    You edit the beats' text, then POST to /synthesize to make the voice."""
    workdir = tempfile.mkdtemp(prefix="dlscr_")
    try:
        video_path = os.path.join(workdir, file.filename or "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        audio_path = os.path.join(workdir, "audio.mp3")
        total_ms = media_duration_ms(video_path)
        if mode == "dub":
            extract_audio(video_path, audio_path)
            beats = transcribe(audio_path)
            if not beats:
                raise HTTPException(422, "No speech detected in the video.")
            translate_segments(beats, target_lang)
        else:                                            # video-aware recap
            beats = make_recap_beats(video_path, total_ms, target_lang)
        return {
            "total_ms": total_ms, "mode": mode, "target_lang": target_lang,
            "beats": [{"i": i, "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                       "text": b.get("speak", "")} for i, b in enumerate(beats)],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/synthesize")
def synthesize(payload: dict = Body(...)):
    """Regenerate voice from an EDITED script.json — no re-transcription needed.
    Body = the (edited) script.json: { total_ms, mode, voice, target_lang, speed, beats:
    [{start_ms, end_ms, text}, ...] }. You edit each beat's `text`, then POST it here."""
    beats = payload.get("beats") or []
    if not beats:
        raise HTTPException(422, "No beats in script.")
    for b in beats:
        b["speak"] = (b.get("text") or "").strip()
    total_ms = int(payload.get("total_ms") or beats[-1].get("end_ms", 0))
    mode = payload.get("mode", "recap")
    speed = float(payload.get("speed", 1.0))
    target_lang = payload.get("target_lang", "Burmese")
    voice = payload.get("voice") or DEFAULT_VOICES.get(target_lang, DEFAULT_VOICE)

    if mode == "dub":
        timeline = build_timeline(beats, total_ms, voice)
    else:
        timeline = build_timeline_recap(beats, total_ms, voice)
    timeline = apply_speed(timeline, speed)

    mp3_buf = io.BytesIO()
    timeline.export(mp3_buf, format="mp3", bitrate="128k")
    srt_text = build_srt(beats, speed)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("output.mp3", mp3_buf.getvalue())
        z.writestr("output.srt", srt_text)
    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="voiceover.zip"'},
    )


@app.post("/process_video")
async def process_video(
    file: UploadFile = File(...),
    voice: str = Form(""),
    target_lang: str = Form("Burmese"),
    mode: str = Form("recap"),
    video_speed: float = Form(1.2),       # how much to speed the VIDEO (only knob)
    voice_speed: float = Form(1.0),       # ignored — voice auto-fits to the video
    original_audio: str = Form("mute"),   # "mute" or "duck"
):
    """Post-ready MP4. ONE knob: video_speed. The voice ALWAYS auto-fits to the sped
    video (sped up just enough to end when the video ends), so it never runs long or
    short — no manual voice speed. e.g. translate recap uses a slower video (0.9)."""
    if not voice:
        voice = DEFAULT_VOICES.get(target_lang, DEFAULT_VOICE)
    vs = max(0.5, min(2.0, video_speed))
    workdir = tempfile.mkdtemp(prefix="dlvid_")
    try:
        video_path = os.path.join(workdir, "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        audio_path = os.path.join(workdir, "audio.mp3")
        total_ms = media_duration_ms(video_path)

        if mode == "dub":
            extract_audio(video_path, audio_path)
            beats = transcribe(audio_path)
            if not beats:
                raise HTTPException(422, "No speech detected in the video.")
            translate_segments(beats, target_lang)
            timeline = build_timeline(beats, total_ms, voice)
        else:                                            # recap
            beats = make_recap_beats(video_path, total_ms, target_lang)
            timeline = build_timeline_recap(beats, total_ms, voice)  # continuous, no stops

        # AUTO-FIT the voice to the (sped) video — no manual voice speed.
        # Only ever speed UP (never slow below natural); cap at 1.8x for listenability.
        video_dur = max(1, int(total_ms / vs))
        fit = len(timeline) / video_dur
        timeline = apply_speed(timeline, max(1.0, min(fit, 1.8)))

        if len(timeline) < 300:                          # < 0.3s => no voice was produced
            raise HTTPException(502, "No voiceover was produced — the recap script or "
                                     "TTS step returned nothing. Check the script/voice step.")

        # durations after speeding (ms)
        video_dur = int(total_ms / vs)
        voice_dur = len(timeline)
        target = max(video_dur, voice_dur)
        if len(timeline) < target:                       # pad voice with trailing silence
            timeline += AudioSegment.silent(duration=target - len(timeline), frame_rate=24000)

        voice_wav = os.path.join(workdir, "voice.wav")
        timeline.export(voice_wav, format="wav")

        # if the voice is longer than the sped video, hold the last video frame
        vpad = ""
        if voice_dur > video_dur:
            vpad = f",tpad=stop_mode=clone:stop_duration={(voice_dur - video_dur) / 1000:.2f}"

        out_mp4 = os.path.join(workdir, "final.mp4")
        if original_audio == "duck":
            fc = (f"[0:v]setpts=PTS/{vs}{vpad}[v];"
                  f"[0:a]atempo={vs},volume=0.12[a0];[1:a]volume=1.0[a1];"
                  f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[a]")
            maps = ["-map", "[v]", "-map", "[a]"]
        else:                                            # mute original
            fc = f"[0:v]setpts=PTS/{vs}{vpad}[v]"
            maps = ["-map", "[v]", "-map", "1:a"]
        _run(["ffmpeg", "-y", "-i", video_path, "-i", voice_wav,
              "-filter_complex", fc, *maps,
              "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", out_mp4])

        with open(out_mp4, "rb") as f:
            data = f.read()
        return StreamingResponse(
            io.BytesIO(data), media_type="video/mp4",
            headers={"Content-Disposition": 'attachment; filename="recap.mp4"'},
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
