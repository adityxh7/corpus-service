cat > README.md <<'EOF'
# Immutable Leakage-Safe Training Corpus Service

FastAPI service implementing the `/build-corpus` endpoint.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000