"""Map URLs to tensorstore kvstore specs (local files, https, s3) and small JSON/bytes helpers."""

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tensorstore as ts


def kvstore_spec(url: str, s3_endpoint: str | None = None) -> dict[str, Any]:
    """Return a tensorstore kvstore spec for a directory URL (always with trailing slash)."""
    url = url.rstrip("/") + "/"
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return {"driver": "http", "base_url": url}
    if parsed.scheme == "s3":
        spec: dict[str, Any] = {"driver": "s3", "bucket": parsed.netloc, "path": parsed.path.lstrip("/")}
        endpoint = s3_endpoint or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            spec["endpoint"] = endpoint
        return spec
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme == "file" else url).expanduser().resolve()
        return {"driver": "file", "path": str(path) + "/"}
    raise ValueError(f"Unsupported URL scheme: {url}")


def join(url: str, *parts: str) -> str:
    return "/".join([url.rstrip("/"), *(p.strip("/") for p in parts)])


def open_kv(url: str, s3_endpoint: str | None = None) -> ts.KvStore:
    return ts.KvStore.open(kvstore_spec(url, s3_endpoint)).result()


def is_access_denied(e: Exception) -> bool:
    """S3 without list permission (e.g. CMEMS ARCO) answers 403 instead of 404 for absent keys."""
    return isinstance(e, ValueError) and "PERMISSION_DENIED" in str(e)


def read_json(kv: ts.KvStore, key: str) -> dict[str, Any] | None:
    try:
        res = kv.read(key).result()
    except ValueError as e:
        if is_access_denied(e):
            return None
        raise
    if res.state != "value":
        return None
    return json.loads(res.value)


def write_json(kv: ts.KvStore, key: str, obj: Any) -> None:
    kv.write(key, json.dumps(obj, indent=1, allow_nan=True).encode()).result()
