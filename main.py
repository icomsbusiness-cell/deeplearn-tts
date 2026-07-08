from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
import edge_tts
import io

app = FastAPI(title="DeepLearn Edge TTS")

# CORS — later replace "*" with your Bolt app domain for safety
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


def fmt_rate(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}%" if v >= 0 else f"{int(v)}%"
    return v


def fmt_pitch(v):
    if isinstance(v, (int, float)):
        return f"+{int(v)}Hz" if v >= 0 else f"{int(v)}Hz"
    return v


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
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        data = buf.getvalue()
        if not data:
            raise HTTPException(status_code=502, detail="no audio returned")
        return Response(content=data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"edge-tts failed: {e}")
