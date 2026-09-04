"""TEMPORARY site-wide password gate.

The site now spends the owner's LLM key (LLM_API_KEY), so access is limited to
whoever knows SITE_PASSWORD. Delete this module and its three hooks in main.py
(middleware, /login routes, the fail-closed check in _server_llm) to return to
the open bring-your-own-key mode preserved on the `byok-public` branch.

Session = one HMAC-derived cookie; changing the password logs everyone out.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from html import escape
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "phantocite_auth"
MAX_AGE_S = 30 * 24 * 3600
# Never gated: the login flow itself, monitoring, static assets, and the
# keyless EduGenAI endpoints (an extension cannot pass a login form and they
# never touch the LLM key; they are rate-limited separately).
OPEN_PATHS = {"/login", "/logout", "/api/health", "/api/verify", "/api/verify_batch",
              "/api/echo", "/favicon.ico"}
OPEN_PREFIXES = ("/static/",)


def password() -> str | None:
    return os.environ.get("SITE_PASSWORD") or None


def gate_enabled() -> bool:
    return password() is not None


def _token(pw: str) -> str:
    return hmac.new(pw.encode(), b"phantocite-session-v1", hashlib.sha256).hexdigest()


def is_authenticated(request: Request) -> bool:
    pw = password()
    if pw is None:
        return True
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), _token(pw))


def check_password(candidate: str) -> bool:
    pw = password()
    return pw is not None and hmac.compare_digest(candidate or "", pw)


def is_open(path: str) -> bool:
    return path in OPEN_PATHS or any(path.startswith(p) for p in OPEN_PREFIXES)


def safe_next(n: str | None) -> str:
    """Only same-origin paths may be redirect targets."""
    return n if n and n.startswith("/") and not n.startswith("//") else "/"


def deny(request: Request):
    path = request.url.path
    if path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"detail": "Password required — sign in at /login."})
    return RedirectResponse(f"/login?next={quote(path)}", status_code=302)


def set_cookie(response, request: Request) -> None:
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, _token(password() or ""), max_age=MAX_AGE_S, httponly=True,
                        secure=secure, samesite="lax", path="/")


def clear_cookie(response) -> None:
    response.delete_cookie(COOKIE, path="/")


def login_page(next_path: str, error: bool = False, status: int = 200) -> HTMLResponse:
    msg = '<p class="err">Wrong password.</p>' if error else ""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Citation Checker · sign in</title>
<style>
  body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#f6f4ef;font:15px/1.5 system-ui,sans-serif;color:#1f2a24}}
  form{{background:#fff;border:1px solid #e3ded4;border-radius:10px;padding:1.6rem 1.8rem;width:min(360px,92vw);box-shadow:0 2px 12px rgba(0,0,0,.05)}}
  h1{{font-size:1.15rem;margin:0 0 .3rem}} p{{margin:.2rem 0 1rem;color:#5c655f;font-size:.9rem}}
  input{{width:100%;box-sizing:border-box;padding:.55rem .7rem;border:1px solid #cfc9bd;border-radius:8px;font:inherit}}
  button{{margin-top:.9rem;width:100%;padding:.6rem;border:0;border-radius:8px;background:#2e6b4f;color:#fff;font:inherit;cursor:pointer}}
  .err{{color:#b3261e;font-weight:600}}
</style></head><body>
<form method="post" action="/login">
  <h1>Citation Checker</h1>
  <p>This instance is password-protected (temporary).</p>
  {msg}
  <input type="hidden" name="next" value="{escape(next_path, quote=True)}">
  <input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
  <button type="submit">Sign in</button>
</form></body></html>"""
    return HTMLResponse(html, status_code=status, headers={
        "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'none'; "
                                   "frame-ancestors 'none'; form-action 'self'; base-uri 'self'",
        "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
    })
