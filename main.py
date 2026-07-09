from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import edge_tts
import io, base64

app = FastAPI(title="DeepLearn Edge TTS")

# CORS — later replace "*" with your Bolt app domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class TTSReq(BaseModel):
    text: str
    voiceId: str = "my-MM-ThihaNeural"
    rate: Union[str, int, float] = "+0%"     # "+40%" or 40
    pitch: Union[str, int, float] = "+0Hz"   # "+0Hz" or 0
    maxChars: int = 32                        # max characters per caption line (short lines)


def fmt_rate(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}%" if v >= 0 else f"{int(v)}%"
    return v


def fmt_pitch(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}Hz" if v >= 0 else f"{int(v)}Hz"
    return v


def ticks_to_srt(ticks: int) -> str:
    # edge-tts offsets are in 100-nanosecond units (10,000,000 per second)
    ms_total = ticks // 10_000
    h = ms_total // 3_600_000
    m = (ms_total % 3_600_000) // 60_000
    s = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


BREAK_CHARS = "။၊.!?…"  # sentence / clause enders -> force a new caption line


def build_srt(boundaries, voice_id: str, max_chars: int) -> str:
    # Burmese has no spaces between words; join without spaces. Latin -> join with spaces.
    burmese = voice_id.startswith("my")
    joiner = "" if burmese else " "

    cues = []            # (start_ticks, end_ticks, text)
    words, start, end = [], None, None

    def flush():
        nonlocal words, start, end
        if words:
            cues.append((start, end, joiner.join(words).strip()))
            words, start, end = [], None, None

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

    out = []
    for i, (st, en, tx) in enumerate(cues, 1):
        out.append(f"{i}\n{ticks_to_srt(st)} --> {ticks_to_srt(en)}\n{tx}\n")
    return "\n".join(out).strip()


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

        srt = build_srt(boundaries, req.voiceId, req.maxChars)
        duration_sec = (boundaries[-1]["offset"] + boundaries[-1]["duration"]) / 10_000_000 if boundaries else 0

        # Return audio + perfectly-synced SRT together (same synthesis = same timing)
        return {
            "audio_base64": base64.b64encode(data).decode("ascii"),
            "srt": srt,
            "duration": round(duration_sec, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"edge-tts failed: {e}")
