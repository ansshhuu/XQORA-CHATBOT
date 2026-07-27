# Testing

Always run tests with `TEST_MODE=true` (the default for `python tests/test_e2e.py --mocked`) to avoid burning API quota; only run the non-mocked, real-API suite (`python tests/test_e2e.py`) deliberately and sparingly.

## Why this matters

`TEST_MODE` (see `app/core/config.py`) gates the real Groq/OpenRouter calls in
`app/services/ai_service.py` and the real Resend call in
`app/services/lead_service.py`. When it's `true`:

- `ai_service.get_ai_response()` never calls Groq/OpenRouter - it raises
  `AIServiceError` instead, which every caller (`intent_agent`, `faq_agent`,
  `recommend_agent`, `lead_agent`) already has tested fallback behavior for.
  This is a safety net *underneath* `tests/test_e2e.py --mocked`'s own
  `unittest.mock.patch` calls, which take precedence when present - it only
  matters for a call site that forgets to mock.
- `lead_service.forward_lead()` never calls Resend - it logs
  `[TEST MODE] would have sent email to ...` and returns `True` instead.

`TEST_MODE` must never be set in production or in Vercel's environment
variables. `check_test_mode_production_safety()` (called at startup in
`app/main.py`) logs a loud error if `TEST_MODE=true` is ever detected
alongside a production-looking `DATABASE_URL` or `VERCEL_ENV=production`, to
catch this accidentally shipping.

## Running the suite

- `python tests/test_e2e.py --mocked` - the default. Mocked AI responses,
  `TEST_MODE=true`, zero real API calls. Safe to run repeatedly.
- `python tests/test_e2e.py` - the real-API suite. Hits Groq/OpenRouter for
  real (paced with a per-request delay to stay under the free-tier rate
  limit) to catch AI-judgment-call regressions the mocked suite can't. Run
  this sparingly, not on every change.
