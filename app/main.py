import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import CORS_ORIGINS
from app.database import init_db
from app.routes import chat

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

class NoCacheStaticFiles(StaticFiles):
    """Static files with browser caching disabled.

    Plain StaticFiles sends ETag/Last-Modified, so browsers cache widget.js
    and revalidate on refresh (a 304 with the old cached copy served) instead
    of always fetching what's actually on disk - confusing during active
    widget development. Remove this override (or scope it to specific paths)
    before a production deploy where caching the widget is actually wanted.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response


app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


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
