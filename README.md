# voice-service

Speech services for Cyclone (RTX 5070 Ti) — STT and TTS over Tailscale.

| Service | Port | Model |
|---------|------|-------|
| Whisper STT | 8100 | `large-v3` |
| Qwen3 TTS | 8200 | `Qwen3-TTS-12Hz-1.7B-CustomVoice` |

## Setup

```bash
git clone git@github.com:ellis0131k-tech/voice-service.git ~/voice-service
```

### Whisper (STT)

```bash
cd ~/voice-service/whisper
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### TTS

```bash
cd ~/voice-service/tts
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # optional, faster attention
```

First TTS run downloads the model from HuggingFace (~3.5 GB).

## Run (dev)

```bash
# Whisper
cd whisper && uvicorn server:app --host 0.0.0.0 --port 8100

# TTS
cd tts && uvicorn server:app --host 0.0.0.0 --port 8200
```

## Install as systemd services

```bash
sudo cp whisper/whisper-api.service /etc/systemd/system/whisper-api@.service
sudo cp tts/tts-api.service /etc/systemd/system/tts-api@.service
sudo systemctl daemon-reload
sudo systemctl enable whisper-api@$USER tts-api@$USER
sudo systemctl start whisper-api@$USER tts-api@$USER
```

## Endpoints

### Whisper — `cyclone:8100`

- `GET /health` — status, model name, device
- `POST /transcribe` — multipart `audio` field, returns `{text, language, duration}`

### TTS — `cyclone:8200`

- `GET /health` — status, model name, device
- `POST /synthesize` — JSON body, returns WAV audio

```bash
# Transcribe
curl -X POST http://cyclone:8100/transcribe -F audio=@recording.wav

# Synthesize
curl -X POST http://cyclone:8200/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}' \
  --output test.wav
```

### Synthesize parameters

| Field | Default | Description |
|-------|---------|-------------|
| `text` | (required) | Text to speak |
| `speaker` | `Ryan` | Voice name (Ryan, Aiden, Vivian, Serena, etc.) |
| `language` | `English` | Language (English, Chinese, Japanese, Korean, etc.) |
| `instruct` | `""` | Emotional/style instruction |

## Environment overrides

- `WHISPER_MODEL` — Whisper model name (default: `large-v3`)
