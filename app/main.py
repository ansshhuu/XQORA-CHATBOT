import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import CORS_ORIGINS, check_required_keys, check_test_mode_production_safety
from app.routes import chat, feedback

logger = logging.getLogger("xqora.main")

app = FastAPI(
    title="XQORA Chatbot API",
    description="Backend API for XQORA Technologies chatbot",
    version="0.1.0",
)

# CORS_ORIGINS defaults to a placeholder domain (see app/core/config.py) — set
# the real production domain(s) via the CORS_ORIGINS env var before deploying.
# allow_credentials=False since the API uses no cookies (session_id travels in
# the request body), which also lets this stay safe even if a wildcard slips in.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup() -> None:
    # Tables are created once, manually, via `python -m app.database` (see its
    # __main__ block) - not here. On serverless (Vercel), this startup event
    # runs on every cold start, so calling Base.metadata.create_all() here
    # would re-run a schema check/DDL against Postgres on every cold start,
    # which is wasteful and risky (racing concurrent cold starts against the
    # same migration). tests/test_e2e.py calls init_db() directly itself for
    # the same reason - this path is deliberately not automatic.
    check_required_keys()
    check_test_mode_production_safety()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last-resort safety net: full detail goes to server logs only, never to
    # the client response (no stack traces, exception text, or API keys).
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again later."})


@app.get("/")
def health_check():
    return {"status": "ok", "message": "XQORA chatbot API is running"}


app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
