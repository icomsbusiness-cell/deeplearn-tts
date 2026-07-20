# deeplearn-tts — timing-synced voiceover service

One FastAPI service that takes a video and returns a Burmese voiceover whose
speech sits at the **same timestamps** as the source, with silence kept where
the source is silent. Personal use — no auth, no database.

## Pipeline
video → ffmpeg audio → Groq Whisper (timestamps) → Gemini 2.5 Flash (translate)
→ Azure Neural TTS per segment → hybrid ±20% timeline assembler → mp3 + srt (zip)

## Environment variables
```
AZURE_SPEECH_KEY      your Azure Speech key
AZURE_SPEECH_REGION   e.g. southeastasia
GROQ_API_KEY          Groq key (Whisper transcription)
GEMINI_API_KEY        Google AI Studio key (translation)
```

## ffmpeg is required
`pydub` and the audio extraction step need `ffmpeg` + `ffprobe` on the host.
On Render, add an **`aptfile`** at the repo root containing:
```
ffmpeg
```
(Render installs apt packages listed there during build.) Locally: `apt install ffmpeg`.

## Run locally
```
pip install -r requirements.txt
uvicorn main:app --reload
```

## Endpoints
- `GET /health` → `{ ok, ffmpeg }`
- `POST /process` (multipart)
  - `file`  — the video
  - `voice` — `my-MM-NilarNeural` (default) or `my-MM-ThihaNeural`
  - `target_lang` — default `Burmese`
  - returns → `voiceover.zip` containing `output.mp3` + `output.srt`

Test:
```
curl -X POST http://localhost:8000/process \
  -F "file=@recap.mp4" -F "voice=my-MM-NilarNeural" -o voiceover.zip
```

## The hybrid ±20% logic (the core idea)
For each speech segment (slot = end − start from the transcript):
1. synthesize at natural rate, measure it
2. fits the slot (within +5%) → keep the natural voice
3. too long → re-synthesize **once** with Azure `<prosody rate>`, capped at **+20%**
4. any residual overflow is absorbed by the silence gaps (cursor never lets a
   clip start before the previous one ends)
Short clips are never stretched to fill — trailing silence is left as-is, so the
output tracks the video's real speech/silence rhythm.

## Notes / limits
- Groq caps upload at ~25MB; audio is extracted mono 16kHz 64k mp3 to stay small.
  For very long sources, swap in AssemblyAI for transcription.
- `/process` is synchronous. For long videos on Render's free tier, consider a
  background job + polling to avoid proxy timeouts (see architecture notes).
- Cost lands on your own Azure account: prebuilt Neural ≈ $16 / 1M chars, with
  0.5M chars/month free (F0). A ~10-min recap is ~10–15k chars.
