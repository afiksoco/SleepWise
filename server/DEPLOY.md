# SleepWise Backend — Deployment Guide

Target: **Railway** (managed Postgres + Docker app, free tier is fine for the POC).

The backend is non-critical to the wake-up path (NFR3) — the alarm fires
on-device even with Wi-Fi and cellular disabled. The server only stores
sessions and serves the weekly rollup.

---

## 1. What ships in this repo

| File | Purpose |
|---|---|
| `Dockerfile` | python:3.12-slim image, installs `server/requirements.txt`, runs uvicorn on `$PORT`. |
| `.dockerignore` | Excludes notebooks, training data, models, videos, the SQLite file. |
| `railway.toml` | Builder = Dockerfile, healthcheck = `GET /`, restart-on-failure. |
| `server/requirements.txt` | Lean server-only deps (fastapi, uvicorn, sqlalchemy, psycopg2-binary). |
| `server/main.py` | FastAPI app, lifespan = init_db. |
| `server/db.py` | SQLAlchemy 2.0 models + engine. `DATABASE_URL` env var; rewrites `postgres://` → `postgresql://` for SQLAlchemy 2.x compat. |
| `server/auth.py` | Bearer-token dep + `POST /devices/register` token mint. |
| `server/logging_config.py` | Stdlib JSON formatter + `X-Request-ID` middleware. |

---

## 2. Railway click-by-click

1. **New project** → **Deploy from GitHub repo**, pick this repo (`SleepWise`).
   - If you'd rather deploy from local CLI: `railway login`, `railway init`, `railway up`.
2. Railway auto-detects the `Dockerfile`. Confirm the **service name** (default is fine).
3. **Add a Postgres database**: project dashboard → **+ New** → **Database** → **PostgreSQL**.
4. **Wire `DATABASE_URL` into the app**: open the app service → **Variables** → **+ Add Reference** → select the Postgres service → variable `DATABASE_URL`. Railway will inject the connection string.
5. **Expose a public URL**: app service → **Settings** → **Networking** → **Generate Domain**. Note the URL — it'll look like `https://sleepwise-production.up.railway.app`.
6. **Redeploy** (Railway does this automatically when env vars change). Watch the build log; on first start you should see:
   ```json
   {"ts":"...","level":"INFO","logger":"sleepwise","msg":"startup_complete"}
   ```

### Env vars

| Name | Source | Required | Notes |
|---|---|---|---|
| `DATABASE_URL` | Postgres service reference | **yes** | Without it, falls back to local SQLite (which is wiped on every redeploy — don't ship like that). |
| `PORT` | Railway-injected | auto | Read by the `CMD`/`startCommand`. |
| `LOG_LEVEL` | optional | no | Default `INFO`. Set to `DEBUG` to verbose. |

You don't need to set anything else for the POC.

---

## 3. Verify the live deployment

Replace `$URL` with your Railway domain.

```bash
# 1. health (no auth)
curl -s $URL/ | jq
# → {"service":"SleepWise Backend","status":"running","sessions_stored":0}

# 2. mint a device
curl -s -X POST $URL/devices/register | jq
# → {"user_id":"<uuid>","token":"<opaque-43-char>"}

# 3. authed upload
TOKEN=...  # paste from step 2
USER=...
curl -s -X POST $URL/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"window_start":"2026-05-11T07:00:00Z","window_end":"2026-05-11T07:30:00Z","started_at":"2026-05-10T23:00:00Z","stages":[]}' | jq

# 4. weekly rollup
curl -s "$URL/sessions/$USER/weekly" -H "Authorization: Bearer $TOKEN" | jq
```

The `X-Request-ID` response header is echoed back on every call — useful for log correlation.

---

## 4. Point the Android app at production

`SleepWiseApi.kt` has `DEFAULT_BASE_URL = "http://10.1.40.24:5000/"` hardcoded.
The Android Claude is wiring a settings screen + DataStore-persisted base URL —
once that lands, paste the Railway URL there. **Use the trailing slash**:
`https://sleepwise-production.up.railway.app/`.

Note: HTTPS removes the need for the `usesCleartextTraffic` workaround
the dev build needed for plain-HTTP `10.0.2.2`.

### First-launch handshake (already implemented client-side by the Android Claude)

1. App launches with no token in DataStore.
2. App calls `POST /devices/register` → receives `(user_id, token)`.
3. App writes both to DataStore.
4. App passes the token to `ApiClient.setAuthToken(token)`.
5. All subsequent requests carry `Authorization: Bearer <token>` via the OkHttp interceptor.

If the user reinstalls or clears app data the token is lost — they'll get a
new `user_id` on next launch. That's the documented limitation. Long-term fix
is real account-based auth, but it's out of scope for the POC.

---

## 5. Notes for the grader / NFR3 demo

- **The wake-up path stays edge-only.** Pull the Railway service down and the
  alarm still fires — this is NFR3-binding behavior.
- **No PII** crosses the wire. `user_id` is a server-minted UUID, not a name
  or email; sleep data is HR/temperature aggregates only.
- **TLS** is handled by Railway's edge (Let's Encrypt managed).
- **Postgres** is managed by Railway; no manual schema migration needed because
  SQLAlchemy creates tables on startup via `init_db()`.

---

## 6. Local dev (still works)

```bash
pip install -r server/requirements.txt
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 5000
```

`DATABASE_URL` unset → SQLite at `server/sleepwise.db`. Same JSON logs, same
auth, same endpoints — only the storage backend differs.

---

## 7. Rollback / wipe

- **Drop all sessions for a user** (authed): `DELETE /sessions/{user_id}`.
- **Wipe everything**: Railway dashboard → Postgres service → **Data** tab →
  drop the `sleep_sessions` and `devices` tables; the app will recreate them
  on next start.
