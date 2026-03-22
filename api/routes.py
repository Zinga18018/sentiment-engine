from fastapi import HTTPException

from .schemas import SingleRequest, BatchRequest, CompareRequest


def register_routes(app, engine):
    """wire up sentiment analysis endpoints."""

    @app.get("/health")
    async def health():
        return engine.health()

    @app.post("/analyze")
    async def analyze(req: SingleRequest):
        if not engine.is_loaded:
            raise HTTPException(503, "model not loaded")
        r = engine.analyze(req.text)
        return {
            "text": r.text, "label": r.label,
            "score": r.score, "inference_ms": r.inference_ms,
        }

    @app.post("/analyze/batch")
    async def analyze_batch(req: BatchRequest):
        if not engine.is_loaded:
            raise HTTPException(503, "model not loaded")
        if len(req.texts) > engine.config.max_batch_size:
            raise HTTPException(400, f"max {engine.config.max_batch_size} texts per batch")

        results, total_ms = engine.analyze_batch(req.texts)
        items = [
            {"text": r.text, "label": r.label, "score": r.score, "inference_ms": r.inference_ms}
            for r in results
        ]
        return {
            "results": items,
            "total_inference_ms": total_ms,
            "avg_inference_ms": round(total_ms / len(results), 1),
            "throughput_texts_per_sec": round(len(results) / (total_ms / 1000), 1),
        }

    @app.post("/analyze/analytics")
    async def analyze_analytics(req: BatchRequest):
        if not engine.is_loaded:
            raise HTTPException(503, "model not loaded")
        if len(req.texts) > engine.config.max_batch_size:
            raise HTTPException(400, f"max {engine.config.max_batch_size} texts per batch")
        return engine.analytics(req.texts)

    @app.post("/compare")
    async def compare(req: CompareRequest):
        if not engine.is_loaded:
            raise HTTPException(503, "model not loaded")
        return engine.compare(req.text_a, req.text_b)
