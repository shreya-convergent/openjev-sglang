"""uv run modal run modal_app.py  (ephemeral B200 smoke test)
uv run modal deploy modal_app.py  (persistent .modal.direct endpoint)
"""

import os
import secrets
from pathlib import Path

import modal

from openjev.profiles import PROFILES

ROOT = Path(__file__).parent
PROFILE_NAME = os.environ.get("OPENJEV_PROFILE", "qwen36")
profile = PROFILES[PROFILE_NAME]
app = modal.App(profile.app_name)
cache = modal.Volume.from_name("openjev-huggingface", create_if_missing=True)
# MODAL_PROXY_TOKEN ("wk-....ws-...") lets the web app reach the proxy-authenticated
# Server. OPENJEV_API_KEY is the shared password callers send to the web app.
proxy_token = modal.Secret.from_name(
    "openjev-sglang-proxy-auth", required_keys=["MODAL_PROXY_TOKEN"]
)
password = modal.Secret.from_name("openjev-sglang-api-key", required_keys=["OPENJEV_API_KEY"])

# GPU memory snapshots skip imports, weight loading, and CUDA graph capture on cold
# starts. SGLang's sleep/wake endpoints exist only in its Python frontend, and the
# weights' CPU backup needs more container memory. OPENJEV_GPU_SNAPSHOT=0 disables.
GPU_SNAPSHOT = os.environ.get("OPENJEV_GPU_SNAPSHOT", "1") == "1"
FRONTEND = os.environ.get("OPENJEV_FRONTEND", "python" if GPU_SNAPSHOT else "rust")
if GPU_SNAPSHOT and FRONTEND != "python":
    raise ValueError("OPENJEV_GPU_SNAPSHOT=1 requires OPENJEV_FRONTEND=python")
SNAPSHOT_SGLANG_ARGS = ["--enable-memory-saver", "--enable-weights-cpu-backup"]
# One small real evaluation, run before each snapshot and after each restore.
WARMUP_REQUEST = (
    b'{"model": "jev-latest", "state": "I was charged twice. Please refund it.",'
    b' "questions": {"refund": {"type": "noul", "instructions": "Refund request?"}}}'
)

# Keep SGLang's tested CUDA environment intact. uv owns a separate, small API
# environment; the package starts SGLang using the base image's Python.
image = (
    modal.Image.from_registry(profile.image)
    .entrypoint([])
    .uv_pip_install("uv==0.12.5")
    .add_local_file(ROOT / "pyproject.toml", "/opt/openjev/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/opt/openjev/uv.lock", copy=True)
    .add_local_file(ROOT / "README.md", "/opt/openjev/README.md", copy=True)
    .add_local_dir(ROOT / "src", "/opt/openjev/src", copy=True)
    .run_commands("cd /opt/openjev && uv sync --frozen --no-default-groups --python 3.12")
    .env(
        {
            "PYTHONPATH": "/opt/openjev/src",
            "HF_HOME": "/cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_CACHE_DIR": "/cache/huggingface/runtime/sglang",
            "TRITON_CACHE_DIR": "/cache/huggingface/runtime/triton",
            "TOKENIZERS_PARALLELISM": "false",
            "OPENJEV_FRONTEND": FRONTEND,
            # Parallel Inductor compile workers can break snapshot creation.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OPENJEV_PROFILE": PROFILE_NAME,
            "OPENJEV_SERVED_MODEL_NAME": os.environ.get(
                "OPENJEV_SERVED_MODEL_NAME", profile.served_model_name
            ),
        }
    )
)


@app.server(
    image=image,
    gpu="B200",
    cpu=8,
    memory=max(profile.memory_mib, 65536) if GPU_SNAPSHOT else profile.memory_mib,
    volumes={"/cache/huggingface": cache},
    port=8000,
    routing_region="us-west",
    compute_region=["us-west", "us-central", "us"],
    min_containers=0,
    max_containers=1,
    target_concurrency=8,
    scaledown_window=300,
    startup_timeout=1200,
    exit_grace_period=30,
    enable_memory_snapshot=GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if GPU_SNAPSHOT else None,
)
class OpenJev:
    # With snapshots on, this runs once per snapshot; restores resume after it.
    @modal.enter(snap=GPU_SNAPSHOT)
    def startup(self):
        from openjev.launch import sleep_backend, start_background

        self.process = start_background(
            "/opt/openjev/.venv/bin/python",
            profile.backend_python,
            sglang_args=SNAPSHOT_SGLANG_ARGS if GPU_SNAPSHOT else None,
        )
        if GPU_SNAPSHOT:
            sleep_backend("http://127.0.0.1:8000", "http://127.0.0.1:30000", WARMUP_REQUEST)

    @modal.enter(snap=False)
    def wake(self):
        from openjev.launch import wake_backend

        if GPU_SNAPSHOT:
            wake_backend("http://127.0.0.1:8000", "http://127.0.0.1:30000", WARMUP_REQUEST)

    @modal.exit()
    def shutdown(self):
        from openjev.launch import stop_background

        if hasattr(self, "process"):
            stop_background(self.process)


# The Server requires Modal proxy auth, and only this CPU-only web app holds the
# token. Callers send the shared password here instead; wrong passwords are
# rejected on CPU, so they never start a B200. The web app hosts the API
# reference, a warm-up route, and a pass-through to the Server.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]>=0.115,<1",
        "httpx>=0.28,<1",
        "jinja2>=3.1.6,<4",
        "orjson>=3.10,<4",
        "pydantic-settings>=2.8,<3",
    )
    .add_local_python_source("openjev")
)
FORWARDED_REQUEST_HEADERS = {"content-type"}
FORWARDED_RESPONSE_HEADERS = {"content-type", "retry-after", "server-timing"}


def _client(url: str, **kwargs):
    import httpx

    return httpx.AsyncClient(base_url=url.rstrip("/"), **kwargs)


async def _server_ready(client) -> bool:
    import httpx

    try:
        response = await client.get("/health", timeout=10)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    if response.status_code in {401, 403}:
        raise PermissionError(f"Server rejected the proxy auth token: {response.text}")
    return response.is_success


def _server_client(url: str, **kwargs):
    return _client(
        url, headers={"Authorization": f"Bearer {os.environ['MODAL_PROXY_TOKEN']}"}, **kwargs
    )


# One poller at a time: repeated /warm calls queue behind it and exit at once.
@app.function(image=web_image, secrets=[proxy_token], timeout=1500, max_containers=1)
async def wait_until_ready():
    """Poll a cold Server (503 while starting) until /health succeeds."""
    import asyncio
    import time

    url = await OpenJev.get_url.aio()
    started = time.monotonic()
    async with _server_client(url) as client:
        while not await _server_ready(client):
            print(f"Server still starting ({time.monotonic() - started:.0f}s); retrying in 5s")
            await asyncio.sleep(5)
    print(f"Server ready after {time.monotonic() - started:.0f}s: {url}")


@app.function(image=web_image, secrets=[proxy_token, password])
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label=f"{profile.app_name}-web")
def web():
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    from openjev.api import create_app
    from openjev.config import Settings

    expected = f"Bearer {os.environ['OPENJEV_API_KEY']}"
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    schema = create_app(Settings()).openapi()
    schema["components"]["securitySchemes"] = {
        "password": {
            "type": "http",
            "scheme": "bearer",
            "description": "The shared OpenJev password.",
        }
    }
    schema["security"] = [{"password": []}]

    def authorized(request: Request) -> bool:
        return secrets.compare_digest(request.headers.get("authorization", ""), expected)

    @web.get("/", include_in_schema=False)
    async def reference():
        return HTMLResponse("""<!doctype html>
<html lang="en">
  <head>
    <title>OpenJev API Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <div id="app"></div>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    <script>
      Scalar.createApiReference('#app', { url: '/openapi.json' });
    </script>
  </body>
</html>""")

    @web.get("/openapi.json", include_in_schema=False)
    async def openapi():
        return schema

    def unavailable(message: str):
        return JSONResponse({"error": {"message": message}}, 503, {"Retry-After": "10"})

    @web.get("/warm", include_in_schema=False)
    async def warm_get():
        raise HTTPException(405, "Use POST /warm with the password", {"Allow": "POST"})

    @web.post("/warm", include_in_schema=False)
    async def warm(request: Request):
        """Return immediately: "ready" if a GPU container is serving, else start one."""
        if not authorized(request):
            raise HTTPException(401, "Missing or invalid password", {"WWW-Authenticate": "Bearer"})
        url = await OpenJev.get_url.aio()
        async with _server_client(url) as client:
            try:
                if await _server_ready(client):
                    return {"status": "ready", "url": url}
            except PermissionError as exc:
                raise HTTPException(502, str(exc)) from exc
        await wait_until_ready.spawn.aio()
        return {"status": "starting", "url": url}

    @web.api_route("/{path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def forward(path: str, request: Request):
        if path not in {"v1/systemone", "v1/models", "v1/limits", "health", "health/live"}:
            raise HTTPException(404)
        if not authorized(request):
            raise HTTPException(401, "Missing or invalid password", {"WWW-Authenticate": "Bearer"})
        headers = {k: v for k, v in request.headers.items() if k in FORWARDED_REQUEST_HEADERS}
        url = await OpenJev.get_url.aio()
        try:
            async with _server_client(url, timeout=150) as client:
                upstream = await client.request(
                    request.method, "/" + path, headers=headers, content=await request.body()
                )
        except httpx.HTTPError as exc:
            # The GPU container went away mid-request (e.g. preemption or scale-down).
            print(f"Forwarding {request.method} /{path} failed: {exc!r}")
            return unavailable("The GPU server restarted during this request. Retry shortly.")
        if upstream.status_code == 503:
            # A cold Server answers 503 until SGLang is up. Modal's idle timer counts from the
            # last request, so keep polling until ready or the Server may scale down right after.
            await wait_until_ready.spawn.aio()
            return unavailable(
                "The GPU server is starting up (a few minutes from cold). Retry shortly."
            )
        return Response(
            upstream.content,
            upstream.status_code,
            {
                k: v
                for k, v in upstream.headers.items()
                if k in FORWARDED_RESPONSE_HEADERS or k.startswith("x-openjev-")
            },
        )

    return web


@app.local_entrypoint()
async def main():
    import json

    from openjev.smoke import smoke_test

    url = await web.get_web_url.aio()
    print(f"OpenJev: {url}")
    result = await smoke_test(url)
    (ROOT / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
