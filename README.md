# Real-Time Sentiment Classification with DistilBERT

Classifies text sentiment using a fine-tuned DistilBERT model (67M parameters). Supports single analysis, batch processing, comparative analysis, and aggregate statistics — all served through a FastAPI REST API.

## how it works

```
text → tokenizer → DistilBERT (67M params) → softmax → POSITIVE/NEGATIVE + confidence score
```

the model is `distilbert-base-uncased-finetuned-sst-2-english`, fine-tuned on the Stanford Sentiment Treebank. it runs through HuggingFace's pipeline abstraction with automatic truncation at 512 tokens.

**what you get back:**
- binary label (POSITIVE / NEGATIVE)
- confidence score (0 to 1)
- inference timing in milliseconds
- batch mode: aggregate stats like positive ratio, average confidence, most extreme texts

## setup

```bash
pip install -r requirements.txt
python main.py
```

runs at `localhost:8003`. swagger docs at `/docs`.

## api

| endpoint | method | what it does |
|----------|--------|-------------|
| `/health` | GET | model status + device info |
| `/analyze` | POST | classify a single text |
| `/analyze/batch` | POST | batch classify up to 100 texts |
| `/analyze/analytics` | POST | batch with aggregate statistics |
| `/compare` | POST | compare sentiment between two texts |

## architecture

```
core/
├── config.py    → model ID, batch limits, all settings
└── engine.py    → SentimentEngine class, batch + compare logic

api/
├── schemas.py   → pydantic request/response models
└── routes.py    → endpoint handlers

main.py          → FastAPI entry point, lifespan
app.py           → streamlit frontend
```

## streamlit demo

```bash
streamlit run app.py
```

type or paste text, see the sentiment label, confidence score, and try batch analysis.

## requirements

- python 3.10+
- ~250MB for model weights
- GPU optional, inference is fast on CPU (~20ms per text)
