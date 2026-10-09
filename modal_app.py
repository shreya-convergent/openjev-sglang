"""Deploy one model per Modal app, each on its own GPU:

uv run modal deploy modal_app.py                               (openjev-sglang: Qwen3.6 on SGLang)
OPENJEV_PROFILE=pplx-decider uv run modal deploy modal_app.py  (pplx-decider: Perplexity's decider)
uv run modal run modal_app.py                                  (smoke-test a deployment)
"""

import os
import secrets
from pathlib import Path

import modal

from openjev.profiles import PROFILES

ROOT = Path(__file__).parent
PROFILE_NAME = os.environ.get("OPENJEV_PROFILE", "qwen36")
# pplx-decider is a decision model with its own readout head and noncausal attention,
# so it runs the checkpoint's bundled autojev server instead of OpenJev on SGLang.
DECIDER = PROFILE_NAME == "pplx-decider"
DECIDER_REPO = "perplexity-ai/pplx-decider-v1.1-27b"
DECIDER_REVISION = "6195aa55a72ae21e503c8a60198f7b390d42adb5"
profile = None if DECIDER else PROFILES[PROFILE_NAME]
APP_NAME = "pplx-decider" if DECIDER else profile.app_name
app = modal.App(APP_NAME)
cache = modal.Volume.from_name("openjev-huggingface", create_if_missing=True)
# MODAL_PROXY_TOKEN ("wk-....ws-...") lets the web app reach the proxy-authenticated
# Server. OPENJEV_API_KEY is the shared password callers send to the web app.
proxy_token = modal.Secret.from_name(
    "openjev-sglang-proxy-auth", required_keys=["MODAL_PROXY_TOKEN"]
)
password = modal.Secret.from_name("openjev-sglang-api-key", required_keys=["OPENJEV_API_KEY"])

# GPU memory snapshots skip imports, weight loading, and kernel compilation on cold
# starts. OPENJEV_GPU_SNAPSHOT=0 disables them.
GPU_SNAPSHOT = os.environ.get("OPENJEV_GPU_SNAPSHOT", "1") == "1"
# SGLang's sleep/wake endpoints exist only in its Python frontend, and the weights'
# CPU backup needs more container memory.
FRONTEND = os.environ.get("OPENJEV_FRONTEND", "python" if GPU_SNAPSHOT else "rust")
if GPU_SNAPSHOT and FRONTEND != "python" and not DECIDER:
    raise ValueError("OPENJEV_GPU_SNAPSHOT=1 requires OPENJEV_FRONTEND=python")
SNAPSHOT_SGLANG_ARGS = ["--enable-memory-saver", "--enable-weights-cpu-backup"]
# One small real evaluation, run before each snapshot and after each restore.
WARMUP_REQUEST = (
    b'{"model": "jev-latest", "state": "I was charged twice. Please refund it.",'
    b' "questions": {"refund": {"type": "noul", "instructions": "Refund request?"}}}'
)
CACHE_ENV = {
    "HF_HOME": "/cache/huggingface",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TRITON_CACHE_DIR": "/cache/huggingface/runtime/triton",
    "TOKENIZERS_PARALLELISM": "false",
    # Parallel Inductor compile workers can break snapshot creation.
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "OPENJEV_PROFILE": PROFILE_NAME,
}

if DECIDER:
    # The checkpoint's pinned inference stack, minus training-only packages.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch==2.14.0",
            "torchvision==0.29.0",
            "transformers==5.17.0",
            "accelerate==1.15.0",
            "pillow==12.3.0",
            "fastapi==0.141.1",
            "uvicorn==0.52.4",
            "httpx==0.28.1",
            "kernels==0.16.1",
            "flash-linear-attention==0.5.2",
            "safetensors==0.8.0",
            "numpy==2.5.3",
            "huggingface-hub==1.31.0",
        )
        .env(CACHE_ENV)
        .add_local_python_source("openjev")
    )
    memory = 65536
    # B200 first; the ~49 GiB BF16 model also fits an H100 when B200s are scarce.
    gpu = ["B200", "H100"]
else:
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
                **CACHE_ENV,
                "PYTHONPATH": "/opt/openjev/src",
                "SGLANG_CACHE_DIR": "/cache/huggingface/runtime/sglang",
                "OPENJEV_FRONTEND": FRONTEND,
                "OPENJEV_SERVED_MODEL_NAME": os.environ.get(
                    "OPENJEV_SERVED_MODEL_NAME", profile.served_model_name
                ),
            }
        )
    )
    memory = max(profile.memory_mib, 65536) if GPU_SNAPSHOT else profile.memory_mib
    # The NVFP4 checkpoint needs Blackwell FP4 support.
    gpu = "B200"


def start_decider():
    """Download the pinned checkpoint (cached in the Volume) and serve it with autojev."""
    import subprocess
    import sys

    from openjev.launch import start_server

    # Download in a child process: hf_xet keeps a log file open in the Volume, and an
    # open file that the snapshot can't find again makes every restore fail.
    download = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download({DECIDER_REPO!r}, revision={DECIDER_REVISION!r}))"
    )
    checkpoint = (
        subprocess.run([sys.executable, "-c", download], check=True, capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1]
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{checkpoint}/source/src",
        "AUTOJEV_CHECKPOINT": checkpoint,
    }
    command = [sys.executable, "-m", "uvicorn", "autojev.server:app", "--host", "0.0.0.0"]
    command += ["--port", "8000"]
    return start_server(command, "http://127.0.0.1:8000/health", "autojev", env=env)


@app.server(
    image=image,
    gpu=gpu,
    cpu=8,
    memory=memory,
    volumes={"/cache/huggingface": cache},
    port=8000,
    routing_region="us-west",
    compute_region=["us-west", "us-central", "us"],
    min_containers=0,
    max_containers=1,
    target_concurrency=8,
    scaledown_window=300,
    startup_timeout=1800,
    exit_grace_period=30,
    enable_memory_snapshot=GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if GPU_SNAPSHOT else None,
)
class OpenJev:
    # With snapshots on, this runs once per snapshot; restores resume after it.
    @modal.enter(snap=GPU_SNAPSHOT)
    def startup(self):
        from openjev.launch import sleep_backend, start_background, verify_evaluation

        if DECIDER:
            self.process = start_decider()
            # Compile kernels before the snapshot so restores don't pay for it.
            verify_evaluation("http://127.0.0.1:8000", WARMUP_REQUEST)
            return
        self.process = start_background(
            "/opt/openjev/.venv/bin/python",
            profile.backend_python,
            sglang_args=SNAPSHOT_SGLANG_ARGS if GPU_SNAPSHOT else None,
        )
        if GPU_SNAPSHOT:
            sleep_backend("http://127.0.0.1:8000", "http://127.0.0.1:30000", WARMUP_REQUEST)

    @modal.enter(snap=False)
    def wake(self):
        from openjev.launch import verify_evaluation, wake_backend

        if not GPU_SNAPSHOT:
            return
        if DECIDER:
            verify_evaluation("http://127.0.0.1:8000", WARMUP_REQUEST)
        else:
            wake_backend("http://127.0.0.1:8000", "http://127.0.0.1:30000", WARMUP_REQUEST)

    @modal.exit()
    def shutdown(self):
        from openjev.launch import stop_background

        if hasattr(self, "process"):
            stop_background(self.process)


# The Server requires Modal proxy auth, and only this CPU-only web app holds the
# token. Callers send the shared password here instead; wrong passwords are
# rejected on CPU, so they never start a GPU. The web app hosts the API
# reference, a warm-up route, and a pass-through to the Server.
web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]>=0.115,<1",
    "httpx>=0.28,<1",
    "jinja2>=3.1.6,<4",
    "orjson>=3.10,<4",
    "pillow>=11,<13",
    "pydantic-settings>=2.8,<3",
)
if DECIDER:
    # autojev's server module imports torch only when loading the model, so its
    # request schema can be built here for the API reference.
    source = f"https://huggingface.co/{DECIDER_REPO}/resolve/{DECIDER_REVISION}/source/src/autojev"
    web_image = web_image.run_commands(
        "mkdir -p /opt/autojev/autojev && touch /opt/autojev/autojev/__init__.py",
        *(
            f'python -c "import urllib.request; urllib.request.urlretrieve('
            f"'{source}/{name}', '/opt/autojev/autojev/{name}')\""
            for name in ("server.py", "types.py")
        ),
    ).env({"PYTHONPATH": "/opt/autojev"})
web_image = web_image.env({"OPENJEV_PROFILE": PROFILE_NAME}).add_local_python_source("openjev")
FORWARDED_PATHS = (
    {"v1/systemone", "v1/models", "health"}
    if DECIDER
    else {"v1/systemone", "v1/models", "v1/limits", "health", "health/live"}
)
FORWARDED_REQUEST_HEADERS = {"content-type"}
FORWARDED_RESPONSE_HEADERS = {"content-type", "retry-after", "server-timing", "x-request-id"}
# How long the web app retries a 529 "busy" answer before passing it on. The decider
# serves one evaluation at a time, so concurrent callers queue here instead.
BUSY_RETRY_SECONDS = 60


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


def _schema():
    if DECIDER:
        from autojev.server import app as autojev_app

        schema = autojev_app.openapi()
        schema["info"]["title"] = "pplx-decider"
    else:
        from openjev.api import create_app
        from openjev.config import Settings

        schema = create_app(Settings()).openapi()
    schema.setdefault("components", {})["securitySchemes"] = {
        "password": {"type": "http", "scheme": "bearer", "description": "The shared password."}
    }
    schema["security"] = [{"password": []}]
    return schema


# One poller at a time: repeated /warm calls queue behind it and exit at once.
@app.function(image=web_image, secrets=[proxy_token], timeout=2400, max_containers=1)
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
@modal.asgi_app(label=f"{APP_NAME}-web")
def web():
    import asyncio
    import time

    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    expected = f"Bearer {os.environ['OPENJEV_API_KEY']}"
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    schema = _schema()

    def authorized(request: Request) -> bool:
        return secrets.compare_digest(request.headers.get("authorization", ""), expected)

    @web.get("/", include_in_schema=False)
    async def reference():
        return HTMLResponse(f"""<!doctype html>
<html lang="en">
  <head>
    <title>{schema["info"]["title"]} API Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <div id="app"></div>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    <script>
      Scalar.createApiReference('#app', {{ url: '/openapi.json' }});
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
        if path not in FORWARDED_PATHS:
            raise HTTPException(404)
        if not authorized(request):
            raise HTTPException(401, "Missing or invalid password", {"WWW-Authenticate": "Bearer"})
        headers = {k: v for k, v in request.headers.items() if k in FORWARDED_REQUEST_HEADERS}
        body = await request.body()
        url = await OpenJev.get_url.aio()
        deadline = time.monotonic() + BUSY_RETRY_SECONDS
        try:
            async with _server_client(url, timeout=150) as client:
                while True:
                    upstream = await client.request(
                        request.method, "/" + path, headers=headers, content=body
                    )
                    if upstream.status_code != 529 or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.25)
        except httpx.HTTPError as exc:
            # The GPU container went away mid-request (e.g. preemption or scale-down).
            print(f"Forwarding {request.method} /{path} failed: {exc!r}")
            return unavailable("The GPU server restarted during this request. Retry shortly.")
        if upstream.status_code == 503:
            # A cold Server answers 503 until it is up. Modal's idle timer counts from the
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
    import asyncio
    import json

    import httpx

    from openjev.smoke import smoke_test

    url = await web.get_web_url.aio()
    print(f"{APP_NAME}: {url}")
    if DECIDER:
        # OpenJev's smoke test asserts OpenJev's 64-answer limit; autojev allows 255.
        headers = {"Authorization": f"Bearer {os.environ['OPENJEV_API_KEY']}"}
        async with httpx.AsyncClient(base_url=url, headers=headers, timeout=150) as client:
            await client.post("/warm")
            result = None
            for _ in range(240):
                response = await client.post("/v1/systemone", content=WARMUP_REQUEST)
                if response.status_code != 503:
                    response.raise_for_status()
                    result = response.json()
                    break
                await asyncio.sleep(10)
        print(json.dumps(result, indent=2))
        return
    result = await smoke_test(url)
    (ROOT / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
