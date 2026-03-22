"""
Real-time Sentiment Engine
===========================
DistilBERT-powered sentiment analysis API.
Supports single text, batch, analytics, and comparison modes.

Usage:
    python main.py
    # Then open http://localhost:8003/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import SentimentEngine, SentimentConfig
from api import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

config = SentimentConfig()
engine = SentimentEngine(config)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.load()
    yield


app = FastAPI(
    title="Real-time Sentiment Engine",
    description="DistilBERT sentiment analysis with batch and analytics support",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app, engine)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=True)
