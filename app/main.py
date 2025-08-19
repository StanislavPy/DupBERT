import json, time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from dupbert.infer import DupBERTService

app = FastAPI(title="DupBERT API", version="1.0.0")
svc = DupBERTService()  # loads config & checkpoint

class PairRequest(BaseModel):
    text_a: str = Field(..., min_length=1)
    text_b: str = Field(..., min_length=1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PairRequest):
    try:
        p, dup = svc.score_pair(req.text_a, req.text_b)
        return {"probability": p, "is_duplicate": dup, "threshold": svc.threshold}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/stream")
def predict_stream(req: PairRequest):
    def gen():
        yield f"data: {json.dumps({'stage': 'start'})}\n\n"
        time.sleep(0.02)
        yield f"data: {json.dumps({'stage': 'encode'})}\n\n"
        p, dup = svc.score_pair(req.text_a, req.text_b)
        yield f"data: {json.dumps({'stage': 'done','probability': p,'is_duplicate': dup,'threshold': svc.threshold})}\n\n"
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
