# Modal + GitHub Pages Plan

This repo already has Modal usage for batch jobs under `modal_run/` and basic Modal setup notes in `README.md`. For a browser-facing app, the clean split is:

- Modal hosts the inference API
- GitHub Pages hosts the static frontend
- The browser calls the Modal API over HTTPS

## Recommended architecture

### Backend

Use a dedicated Modal app for web inference instead of reusing the current job-style scripts.

Suggested file:

- `modal_run/web_api.py`

Recommended stack:

- `FastAPI` for routes and request validation
- `modal.Image` for dependencies
- `@modal.asgi_app()` for HTTP serving
- `CORSMiddleware` to allow the GitHub Pages origin

Start simple with these routes:

- `GET /health`
- `POST /generate`
- `GET /docs`

If you want OpenAI-compatible clients later, add a separate compatibility route instead of forcing the frontend to speak the OpenAI schema on day one.

### Frontend

Host a plain static site on GitHub Pages.

Suggested directory:

- `frontend/`

Suggested files:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
- `frontend/config.js`

`config.js` should contain only public values, such as:

- Modal API base URL
- UI defaults like max tokens or temperature

Do not put private secrets in the frontend.

## Proposed API contract

Keep the browser contract minimal and stable:

### `POST /generate`

Request:

```json
{
  "prompt": "Explain this chart in one paragraph",
  "system": "You are a concise analyst.",
  "max_tokens": 300,
  "temperature": 0.4
}
```

Response:

```json
{
  "text": "The chart shows...",
  "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
  "request_id": "uuid-or-timestamp"
}
```

This is easier to wire into plain JavaScript than a full chat-completions schema. If you later need streaming, add:

- `POST /generate/stream`

and return Server-Sent Events.

## Modal-specific decisions

The Modal example you linked is a good base because it already targets:

- Open internet access
- vLLM serving
- fast boot / cold-start tuning

For your first version:

- Prefer one small dedicated model
- Optimize for reliability over maximum throughput
- Use the example's fast-boot settings if you expect scale-to-zero
- Mount a Modal Volume for model cache so weights are not fetched on every cold start

If your workload is bursty and user-facing, start with quick cold start. If usage becomes steady, revisit the tuning for better token throughput.

## CORS and browser access

Your Modal API must explicitly allow your GitHub Pages origin, for example:

- `https://<username>.github.io`
- `https://<username>.github.io/<repo>`

In development, also allow:

- `http://127.0.0.1:5500`
- `http://localhost:5500`

Do not leave CORS wide open once the real site is live unless the API is intentionally public.

## Auth and abuse protection

GitHub Pages cannot safely hide server secrets, so choose one of these patterns:

1. Public demo API with strict rate limits and narrow capabilities
2. Modal API protected by simple app-level auth for invited users
3. Separate lightweight auth/proxy layer in front of the model API

For an MVP, option 1 is simplest if:

- requests are cheap
- the model is low-risk
- you are comfortable with public access

If not, add backend auth before launch.

## Deployment sequence

1. Build the Modal API locally and verify `GET /health`
2. Deploy the Modal app and confirm the public URL
3. Test `POST /generate` with `curl`
4. Build the static frontend against that URL
5. Deploy the frontend with GitHub Pages
6. Restrict CORS to the final Pages origin
7. Add basic logging and request IDs

## Suggested implementation order in this repo

1. Create `modal_run/web_api.py`
2. Add `fastapi` to project dependencies and install `vllm` in the Modal image
3. Create `frontend/`
4. Add a GitHub Pages workflow under `.github/workflows/`
5. Add frontend deployment notes to `README.md`

## Good first cut

If we want the least risky first milestone, build this:

- Modal endpoint: `POST /generate`
- Static page with one text input, one submit button, and one output area
- No streaming
- No login
- Simple rate limiting or narrow public use

That gets us a real end-to-end demo quickly, and we can add streaming, nicer UI, and stronger auth after the pipeline is working.
