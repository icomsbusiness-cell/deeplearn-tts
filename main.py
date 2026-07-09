from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import edge_tts
import io, base64, re

app = FastAPI(title="DeepLearn Edge TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class TTSReq(BaseModel):
    text: str
    voiceId: str = "my-MM-ThihaNeural"
    rate: Union[str, int, float] = "+0%"
    pitch: Union[str, int, float] = "+0Hz"
    maxChars: int = 28


def fmt_rate(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}%" if v >= 0 else f"{int(v)}%"
    return v


def fmt_pitch(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}Hz" if v >= 0 else f"{int(v)}Hz"
    return v


def fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


BREAK_CHARS = set("။၊!?….")


def build_srt_from_boundaries(boundaries, voice_id: str, max_chars: int) -> tuple[str, float]:
    """Build SRT from WordBoundary events (accurate timing)."""
    burmese = voice_id.startswith("my")
    joiner = "" if burmese else " "

    cues = []
    words, start, end = [], None, None

    def flush():
        nonlocal words, start, end
        if words:
            cues.append((start / 10_000_000, end / 10_000_000, joiner.join(words).strip()))
            words.clear()
            start = None
            end = None

    for b in boundaries:
        word = b.get("text", "")
        off = b.get("offset", 0)
        dur = b.get("duration", 0)
        if start is None:
            start = off
        words.append(word)
        end = off + dur
        line_len = len(joiner.join(words))
        ends_sentence = word and word[-1] in BREAK_CHARS
        if ends_sentence or line_len >= max_chars:
            flush()
    flush()

    if not cues:
        return "", 0.0

    duration = cues[-1][1]
    out = []
    for i, (st, en, tx) in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_srt_time(st)} --> {fmt_srt_time(en)}\n{tx}\n")
    return "\n".join(out).strip(), duration


def build_srt_from_text(text: str, duration: float, max_chars: int, voice_id: str) -> str:
    """Fallback SRT: split text proportionally when no WordBoundary data."""
    burmese = voice_id.startswith("my")

    # Split into caption chunks
    chunks = []
    current = ""
    for char in text:
        current += char
        if char in BREAK_CHARS or len(current) >= max_chars:
            stripped = current.strip()
            if stripped:
                chunks.append(stripped)
            current = ""
    if current.strip():
        chunks.append(current.strip())

    if not chunks:
        chunks = [text]

    total_chars = sum(len(c) for c in chunks) or 1
    elapsed = 0.0
    out = []
    for i, chunk in enumerate(chunks, 1):
        frac = len(chunk) / total_chars
        start = elapsed
        end = elapsed + frac * duration
        elapsed = end
        out.append(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{chunk}\n")

    return "\n".join(out).strip()


def estimate_duration_from_rate(text: str, rate_str: str) -> float:
    """Estimate TTS duration based on text length and speed rate."""
    # Base: ~5 chars/second for Burmese neural voice at 0% rate
    base_chars_per_sec = 5.0
    # Parse rate like "+40%" or "-20%"
    rate_val = 0
    try:
        rate_val = int(re.sub(r'[^-\d]', '', str(rate_str)))
    except Exception:
        pass
    speed_factor = 1.0 + (rate_val / 100.0)
    speed_factor = max(0.5, min(2.0, speed_factor))
    chars = len(text.replace(" ", ""))
    return chars / (base_chars_per_sec * speed_factor)


@app.get("/")
def health():
    return {"ok": True, "service": "edge-tts"}


@app.post("/tts")
async def tts(req: TTSReq):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    try:
        communicate = edge_tts.Communicate(
            text=req.text,
            voice=req.voiceId,
            rate=fmt_rate(req.rate),
            pitch=fmt_pitch(req.pitch),
        )
        audio = io.BytesIO()
        boundaries = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                boundaries.append(chunk)

        data = audio.getvalue()
        if not data:
            raise HTTPException(status_code=502, detail="no audio returned")

        # Try WordBoundary-based SRT first (accurate)
        srt, duration = build_srt_from_boundaries(boundaries, req.voiceId, req.maxChars)

        # Fallback: proportional SRT from text if no boundaries (e.g. Burmese voice)
        if not srt or duration == 0:
            duration = estimate_duration_from_rate(req.text, req.rate)
            srt = build_srt_from_text(req.text, duration, req.maxChars, req.voiceId)

        return {
            "audio_base64": base64.b64encode(data).decode("ascii"),
            "srt": srt,
            "duration": round(duration, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"edge-tts failed: {e}")
