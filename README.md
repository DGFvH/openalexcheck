# openalexcheck — citation hallucination & misquote checker

A small web tool for checking the references in a student paper.

You upload a **PDF or DOCX** and paste **your own LLM API key** (Claude,
ChatGPT, or Gemini) — the key is used for this one analysis only, sent
directly to the provider, and **never stored or logged**. Two checks can be
ticked:

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

OpenAlex needs no API key. The LLM runs on the key you paste in the form.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Then open <http://localhost:8000>.

Optional: set `OPENALEX_MAILTO=you@example.com` to use OpenAlex's polite pool
(faster, more reliable rate limits).

## Providers and default models

| Provider dropdown | Default model      | Key type            |
|-------------------|--------------------|---------------------|
| Claude (Anthropic)| `claude-opus-4-8`  | `sk-ant-...`        |
| Gemini (Google)   | `gemini-2.5-flash` | AI Studio API key   |
| ChatGPT (OpenAI)  | `gpt-4o-mini`      | `sk-...`            |

A different model can be typed into the optional *Model* field.

## Privacy notes

- The API key is sent with the analysis request, held in memory for the
  duration of that request, and discarded. It is never written to disk,
  never logged, and never echoed back to the browser.
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

## Tests

```bash
pip install pytest
pytest
```
