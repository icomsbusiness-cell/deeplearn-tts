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
import shutil
import zipfile
import tempfile
import subprocess
from typing import List, Dict

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# ---------------------------------------------------------------- config

AZURE_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "southeastasia")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = "gemini-2.5-flash"          # not lite, not 1.5 (deprecated)
GROQ_MODEL = "whisper-large-v3"
DEFAULT_VOICE = "my-MM-NilarNeural"        # or my-MM-ThihaNeural

# Hybrid timing knob: how much we are allowed to speed a clip up to fit its slot.
MAX_SPEEDUP = 0.20                         # +20%
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

def transcribe(audio_path: str) -> List[Dict]:
    """Groq Whisper -> [{start_ms, end_ms, text}] with segment timestamps."""
    with open(audio_path, "rb") as f:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (os.path.basename(audio_path), f, "audio/mpeg")},
            data={
                "model": GROQ_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=600,
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Groq transcription failed: {r.text[:300]}")
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


# ---------------------------------------------------------------- translation

def translate_segments(segments: List[Dict], target: str = "Burmese") -> None:
    """Translate all segments in ONE Gemini call so count/order is preserved."""
    if not segments:
        return
    numbered = [{"i": i, "text": s["text"]} for i, s in enumerate(segments)]
    prompt = (
        f"Translate each item's 'text' into natural spoken {target}. "
        "Keep it concise and speakable (this is for a voiceover). "
        "Return ONLY a JSON array of objects like {\"i\": 0, \"t\": \"...\"}, "
        "same length and order, no markdown, no commentary.\n\n"
        + json.dumps(numbered, ensure_ascii=False)
    )
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
        },
        timeout=300,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini translation failed: {r.text[:300]}")
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    items = json.loads(raw)
    by_i = {int(it["i"]): it.get("t", "") for it in items}
    for i, s in enumerate(segments):
        s["translated"] = by_i.get(i, s["text"]).strip()


# ---------------------------------------------------------------- Azure TTS

def _ssml(text: str, voice: str, rate_pct: int) -> str:
    rate = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xml:lang='my-MM'>"
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
    return AudioSegment.from_file(io.BytesIO(r.content), format="wav")


def synth_segment_hybrid(text: str, slot_ms: int, voice: str) -> AudioSegment:
    """
    HYBRID +-20% strategy:
      1. synthesize at natural rate, measure it
      2. if it fits the slot (within +5%): keep as-is (natural voice)
      3. if it's too long: re-synthesize ONE time with Azure rate, capped at +20%
         (Azure changes tempo without wrecking pitch — better than ffmpeg atempo)
      4. residual overflow is absorbed later by the silence gaps (cursor logic)
    Short clips are never stretched to fill — we just leave trailing silence.
    """
    clip = azure_tts(text, voice, 0)
    if slot_ms <= 0 or len(clip) <= slot_ms * (1 + SLACK):
        return clip
    needed = len(clip) / slot_ms - 1.0          # e.g. 0.35 => 35% too long
    rate_pct = int(round(min(needed, MAX_SPEEDUP) * 100))
    if rate_pct <= 0:
        return clip
    return azure_tts(text, voice, rate_pct)


# ---------------------------------------------------------------- assembler

def build_timeline(segments: List[Dict], total_ms: int, voice: str) -> AudioSegment:
    """Silence base the length of the video; overlay each clip at its start time."""
    base = AudioSegment.silent(duration=total_ms, frame_rate=24000)
    cursor = 0
    for s in segments:
        slot = s["end_ms"] - s["start_ms"]
        clip = synth_segment_hybrid(s["translated"], slot, voice)
        s["tts_dur_ms"] = len(clip)
        place_at = max(s["start_ms"], cursor)   # never overlap the previous clip
        base = base.overlay(clip, position=place_at)
        cursor = place_at + len(clip)
    return base


def build_srt(segments: List[Dict]) -> str:
    def ts(ms):
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{ts(s['start_ms'])} --> {ts(s['end_ms'])}")
        lines.append(s.get("translated", s["text"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg"))}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    voice: str = Form(DEFAULT_VOICE),
    target_lang: str = Form("Burmese"),
):
    """Full pipeline. Returns a zip containing output.mp3 + output.srt."""
    workdir = tempfile.mkdtemp(prefix="dltts_")
    try:
        video_path = os.path.join(workdir, file.filename or "input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        audio_path = os.path.join(workdir, "audio.mp3")
        extract_audio(video_path, audio_path)

        total_ms = media_duration_ms(video_path)
        segments = transcribe(audio_path)
        if not segments:
            raise HTTPException(422, "No speech detected in the video.")

        translate_segments(segments, target_lang)
        timeline = build_timeline(segments, total_ms, voice)

        mp3_buf = io.BytesIO()
        timeline.export(mp3_buf, format="mp3", bitrate="128k")
        srt_text = build_srt(segments)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("output.mp3", mp3_buf.getvalue())
            z.writestr("output.srt", srt_text)
        zip_buf.seek(0)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="voiceover.zip"'},
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
