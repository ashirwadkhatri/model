"""Sentiment service for general YouTube comments (trained model + trained Hinglish detector).

Run with run_service.bat. Works with a two-class model (Negative/Positive, e.g. model_yt) or a
three-class model (Negative/Neutral/Positive, e.g. model_yt3): the class names are read from the model's config.json.

  GET  /health                         -> {"ok": true, ...}
  POST /predict  {"texts": [...]}      -> per-text label + probabilities
  POST /predict_batch {"texts": [...]} -> {"results":[{"label","confidence","prob_positive", ...}]}
  POST /analyze  {"comments": [...]}   -> overall counts
Skipped comments (too short, non-Latin script, Hinglish) come back with confidence 0.0 and prob_positive null.
Their label is "Neutral" with a two-class model, and "Skipped" with a three-class model (where "Neutral" is a real class).
Requests need the header  X-API-Key: <SENTIMENT_API_KEY>  (SENTIMENT_API_KEY).

On CPU the model is quantized to int8 (smaller and faster). Set QUANTIZE=0 to turn that off.
"""
import argparse, hmac, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hinglish_detector import available as hinglish_available, flags as hinglish_flags, threshold as hinglish_threshold
from preprocess import clean_text, non_latin_ratio

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model_yt3")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8200)), help="Render sets PORT automatically")
ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="use 0.0.0.0 on Render")
ap.add_argument("--max_len", type=int, default=128, help="must match training (--max_len)")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--threshold", type=float, default=None, help="two-class models only: Positive if prob_positive >= this (default: the value saved in <model>/threshold.json, else 0.5)")
args = ap.parse_args()
if args.threshold is None:
    try:
        args.threshold = float(json.load(open(os.path.join(args.model, "threshold.json")))["positive_threshold"])
    except Exception:
        args.threshold = 0.5

KEY = os.environ.get("SENTIMENT_API_KEY", "")
MAX_ITEMS, MAX_BODY = 500, 2_000_000
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 1)))   # small Render CPUs: fewer threads, less memory
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
QUANTIZED = False
if dev == "cpu" and os.environ.get("QUANTIZE", "1") == "1":          # CPU only; on a GPU the model stays as it is
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    QUANTIZED = True
lock = threading.Lock()

# class names come from the model itself (config.json -> id2label), so the order is never assumed
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]
POS_I, NEG_I = LABELS.index("Positive"), LABELS.index("Negative")
NEU_I = LABELS.index("Neutral") if "Neutral" in LABELS else None
THREE = NEU_I is not None
SKIP_LABEL = "Skipped" if THREE else "Neutral"

print(f"hinglish_detector={'on threshold=' + str(hinglish_threshold()) if hinglish_available() else 'OFF (hinglish_model.joblib not found)'}", flush=True)
print(f"model={args.model} classes={LABELS} device={dev} int8={QUANTIZED} "
      f"{'decision=most likely class' if THREE else f'threshold={args.threshold}'} key_required={bool(KEY)}", flush=True)


@torch.no_grad()
def probs(cleaned):
    """Probability vector (one value per class) for each cleaned text; None for texts too short, non-English or Hinglish."""
    out = [None] * len(cleaned)
    idx = [i for i, t in enumerate(cleaned) if len(t) >= 3 and non_latin_ratio(t) <= 0.3]   # 1) script check (no word list)
    if hinglish_available() and idx:                                                    # 2) trained Hinglish detector
        hf = hinglish_flags([cleaned[i] for i in idx])
        idx = [i for i, f in zip(idx, hf) if not f]
    for s in range(0, len(idx), args.batch):
        chunk = idx[s:s + args.batch]
        enc = tok([cleaned[i] for i in chunk], padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt").to(dev)
        with lock:
            p = torch.softmax(model(**enc).logits.float(), -1).cpu().tolist()
        for i, v in zip(chunk, p):
            out[i] = v
    return out


def label(v):
    if v is None:
        return None
    if THREE:
        return LABELS[max(range(len(v)), key=v.__getitem__)]
    return "Positive" if v[POS_I] >= args.threshold else "Negative"


def confidence(v):
    return max(v) if THREE else max(v[POS_I], 1 - v[POS_I])


def extra(v):
    """Probabilities to send along with the label."""
    d = {"prob_positive": None if v is None else round(v[POS_I], 4)}
    if THREE:
        d["prob_neutral"] = None if v is None else round(v[NEU_I], 4)
        d["prob_negative"] = None if v is None else round(v[NEG_I], 4)
    return d


def predict(texts):
    cleaned = [clean_text(t) for t in texts]
    return [{"label": label(v), **extra(v)} for v in probs(cleaned)]


def predict_batch(texts):
    """Format used by cache-service/analysis.py and the YouTube app: one item per input, same order.
    Skipped texts get confidence 0.0 and prob_positive null (label "Neutral" for two-class models, "Skipped" for three-class)."""
    out = []
    for v in probs([clean_text(t) for t in texts]):
        if v is None:
            out.append({"label": SKIP_LABEL, "confidence": 0.0, **extra(None)})
        else:
            out.append({"label": label(v), "confidence": round(confidence(v), 4), **extra(v)})
    return out


def pct(pos, neg):
    return round(100 * pos / (pos + neg), 1) if pos + neg else None


def analyze(comments):
    overall = [label(v) for v in probs([clean_text(c) for c in comments])]
    pos, neg, neu = overall.count("Positive"), overall.count("Negative"), overall.count("Neutral")
    res = {"positive": pos, "negative": neg, "scored": pos + neg + neu, "positive_pct": pct(pos, neg)}
    if THREE:
        res["neutral"] = neu
    return {"overall": res}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        return not KEY or hmac.compare_digest(self.headers.get("X-API-Key", ""), KEY)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True, "model": args.model, "classes": LABELS, "device": dev,
                                   "int8": QUANTIZED, "threshold": None if THREE else args.threshold})
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authed():
            return self.send(401, {"error": "Bad or missing X-API-Key"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self.send(413, {"error": "Body too large"})
            data = json.loads(self.rfile.read(n) or b"{}")
            field = "texts" if self.path in ("/predict", "/predict_batch") else "comments" if self.path == "/analyze" else None
            if field is None:
                return self.send(404, {"error": "Not found"})
            items = data.get(field)
            if not isinstance(items, list) or len(items) > MAX_ITEMS or not all(isinstance(x, str) for x in items):
                return self.send(400, {"error": f"'{field}' must be a list of up to {MAX_ITEMS} strings"})
            if self.path == "/predict_batch":
                self.send(200, {"results": predict_batch(items)})
            else:
                self.send(200, {"results": predict(items)} if field == "texts" else analyze(items))
        except Exception as e:                      # never crash the server on a bad request
            print("error:", repr(e), flush=True)
            self.send(500, {"error": "Internal error"})


if __name__ == "__main__":
    print(f"Sentiment service on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()