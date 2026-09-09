# openalexcheck — citation hallucination & misquote checker

**Live:** [phantocite.com](https://www.phantocite.com) · **Stack:** FastAPI · streaming NDJSON · provider-agnostic LLM layer (Claude / GPT / Gemini) · OpenAlex API · vanilla-JS SPA (no build step)

A small web tool for checking the references in a student paper.

> **Temporary mode (current `main`).** The site runs on the owner's LLM key
> (`LLM_API_KEY` / `LLM_PROVIDER` in the server environment) and is behind a
> simple password (`SITE_PASSWORD`); the form has no key field. The original
> bring-your-own-key version is preserved on the `byok-public` branch. The
> keyless EduGenAI endpoints (`/api/verify*`, `/api/echo`) are not gated.

You upload a **PDF or DOCX**; the analysis runs on the configured LLM (Claude,
ChatGPT, or Gemini). Two checks can be ticked:

1. **Hallucination check** — every entry in the reference list is looked up in
   [OpenAlex](https://openalex.org) (by DOI when present, otherwise by fuzzy
   title/author/year matching). References that cannot be found anywhere are
   flagged as *potential hallucinations*. References that *almost* match —
   e.g. a correct DOI attached to the wrong title — are flagged as *fuzzy
   matches*, with the candidate works listed on a second screen so you can
   pick the right one.

2. **Misquote check** — the LLM locates the sentence(s) where each source is
   cited in the body (plus one or two surrounding sentences) and compares them
   with the abstract retrieved from OpenAlex. You get a side-by-side view of
   the student's citation context and the real abstract, with a verdict:
   *consistent*, *likely mismatch*, *mismatch*, or *unclear*. So if a paper is
   about macroeconomic productivity but the student cites it as if it were
   about labour productivity, it gets flagged.

## How it works

```
upload ──▶ text extraction (pypdf / python-docx)
       ──▶ LLM pass 1: extract bibliography entries + in-text citation contexts
       ──▶ OpenAlex: resolve each reference (DOI lookup, title search, scoring)
       ──▶ LLM pass 2: compare citation contexts against OpenAlex abstracts
       ──▶ UI: results screen + fuzzy-matches screen
```

The analysis endpoint **streams progress and results** as newline-delimited
JSON. The browser shows a live progress bar and log, and reference cards
appear one-by-one as each is resolved (misquote verdicts fill in afterwards).
Besides the nicer UX, streaming keeps the connection alive with periodic
heartbeats, so a long analysis can't be dropped by an idle-connection timeout
(the usual cause of a browser "Failed to fetch").

OpenAlex needs no API key (the site's `OPENALEX_MAILTO` puts it in the polite
pool). The LLM runs on the server's key.

**Downloads.** Results export as CSV, Markdown, HTML or PDF. The first three
are written in the browser; the PDF is typeset by `POST /api/report.pdf`
(reportlab, `app/report.py`) from the same rows and streamed back — nothing is
written to disk.

**Capacity.** The output-token cap per LLM call is fixed server-side
(`LLM_MAX_TOKENS`, default 12000) and sized to the analysis time budget
(`ANALYSIS_BUDGET_S`, default 270 s): that comfortably covers a typical student
paper of about 25 pages with around 30 references. Longer documents are
analysed as far as the budget allows and the result is marked as partial.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Then open <http://localhost:8000>.

Optional: set `OPENALEX_MAILTO=you@example.com` to use OpenAlex's polite pool
(faster, more reliable rate limits).

## Deploying

Production runs on Vercel (zero-config FastAPI, entry point `app/main.py`),
auto-deployed from `main`. Things the repo cannot set for you:

- **Environment variables** (Vercel → Settings → Environment Variables):
  `SITE_PASSWORD` (the temporary gate; without it the key-spending endpoints
  refuse to run), `LLM_API_KEY` (the owner's provider key), `LLM_PROVIDER`
  (`anthropic` default, or `openai` / `gemini`), and
  `OPENALEX_MAILTO` — required in production (polite pool; the app logs a
  warning without it). Optional tuning: `ANALYSIS_BUDGET_S` (wall-clock budget
  per analysis, default 270; the run stops starting new work and reports
  `incomplete` in its final event), `OPENALEX_RPS` (request pacing, default 8),
  `RESOLVE_WORKERS` (parallel lookups per analysis, default 4),
  `LLM_READ_TIMEOUT_CAP_S` (OpenAI/Gemini read-timeout cap, default 280),
  `LLM_MAX_TOKENS` (output-token cap per LLM call, default 12000).
- **Function max duration** (Vercel → Settings → Functions): set it above
  `ANALYSIS_BUDGET_S` + 30 s so the app's own budget always wins over the
  platform kill (which ends the stream silently).
- **Rate limiting**: the app throttles `/api/*` per client IP in-process
  (best-effort per instance); add a Vercel Firewall rate-limit rule for a
  durable control.
- **After a deploy**: `GET /api/health` shows the live `api_version` and git
  sha; bump `API_VERSION` in `app/main.py` with every deployed change.

## Providers and default models

| Provider dropdown | Default model      | Key type            |
|-------------------|--------------------|---------------------|
| Claude (Anthropic)| `claude-sonnet-5`  | `sk-ant-...`        |
| Gemini (Google)   | `gemini-2.5-flash` | AI Studio API key   |
| ChatGPT (OpenAI)  | `gpt-4o-mini`      | `sk-...`            |

A different model can be typed into the optional *Model* field.

## Privacy & key safety

Both keys (LLM and optional OpenAlex) are strictly one-time use:

- They arrive in the POST body, are held in memory only for the duration of
  that request, and are discarded. Nothing is written to disk and there is no
  database or cache.
- Keys never appear in this app's own URLs, and request bodies are never
  logged. (Access logs do record query strings — which is why the keys travel
  in the body or the `X-OpenAlex-Key` header — and the httpx request-line
  logger, which would print the OpenAlex `api_key` parameter, is silenced.)
- Cookies: signing in sets one session cookie. Google Analytics (gtag.js)
  runs in Consent Mode — its cookies are set once the visitor clicks "Accept
  cookies" in the bar (remembered in localStorage). Analytics only ever sees
  page views and usage, never document text, references, keys or results.
- `GET /api/health` reports the live build (`api_version`, git sha) and whether
  OpenAlex requests go through the polite pool (`OPENALEX_MAILTO` set).
- Every error message that leaves the server passes through a redaction
  helper (`app/keysafety.py`) that strips the key strings. This matters
  because httpx embeds full request URLs — query string included — in its
  exception text, and provider error bodies are quoted in error details.
- FastAPI's default 422 validation response echoes request input back; the
  app overrides that handler to return field locations only.
- The key inputs use `type="password"` with `autocomplete="new-password"`, so
  browsers mask them and don't offer to save them. The page keeps them in the
  form only so the fuzzy-match screen can run follow-up comparisons; a reload
  clears them.
- The document text is sent to the LLM provider you selected (that is what
  the key is for) and reference titles/DOIs are sent to OpenAlex. Nothing is
  stored server-side.

## Limitations

- Scanned PDFs without a text layer need OCR first.
- OpenAlex does not have an abstract for every work; those references get an
  *unclear* misquote verdict.
- The misquote check compares against the **abstract** only — a claim that is
  supported by the full text but not visible in the abstract can be flagged
  as unclear or a likely mismatch. Treat verdicts as leads for a human
  reviewer, not as final judgements.

## Use it inside EduGenAI (extension)

This app doubles as a backend for an EduGenAI (or any function-calling)
extension. Two keyless JSON endpoints wrap the OpenAlex verification — no LLM
runs server-side, so the extension needs no API key:

- `POST /api/verify` — verify one reference; returns existence, a field-by-field
  metadata comparison (title, authors, year, journal, DOI, volume, issue,
  pages), and the abstract.
- `POST /api/verify_batch` — verify a whole bibliography in one call (max 200).

The calling platform's own model extracts references from the paper and does
the misquote reasoning with the returned abstracts. An optional OpenAlex
Premium key can be supplied via the `X-OpenAlex-Key` header (ideal for the
extension's secure/Key-Vault header store).

Step-by-step setup instructions (with the exact function schema and a
downloadable PDF) are served by the app itself at **`/edugenai`**, and linked
from the main page. To regenerate the PDF after editing the instructions:

```bash
pip install -r requirements-dev.txt
python scripts/build_edugenai_pdf.py
```

## Tests

```bash
pip install pytest
pytest
```

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE)
© Daniël van Hemert.

You are free to use, modify, self-host, and share it for any **noncommercial**
purpose — personal, academic, research, or non-profit. Commercial use
(including selling it or offering it as a paid service) is **not** granted by
this license; contact the author for commercial terms.

Data sources retain their own terms: bibliographic metadata and abstracts come
from [OpenAlex](https://openalex.org) (CC0), and analysis runs on whichever LLM
provider's key you supply.
