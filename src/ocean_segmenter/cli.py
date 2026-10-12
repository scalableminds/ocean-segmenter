"""CLI generated from the Pydantic request model (via pydantic-settings).

    ocean-seg segment --source_url URL --min 34 --max 35 --output_url out/so.zarr
    ocean-seg view out/so.zarr
"""

import functools
import http.server
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

from . import neuroglancer, pipeline
from .models import SegmentRequest


class Segment(SegmentRequest):
    """Run the segmentation pipeline and print the result JSON."""

    def cli_cmd(self) -> None:
        result = pipeline.run(SegmentRequest.model_validate(self.model_dump()))
        print(result.model_dump_json(indent=2, exclude={"segments"}))


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def _bind(port: int, handler) -> http.server.ThreadingHTTPServer:
    """Bind 127.0.0.1:port, or a free port if it is taken (0 = always pick a free one)."""
    try:
        return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        if port == 0:
            raise
        logging.warning("Port %d is in use, picking a free one", port)
        return http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)


class View(BaseModel):
    """Print a Neuroglancer link for an output group. Local paths are served over HTTP with CORS."""

    output: CliPositionalArg[str] = Field(description="Output group: local path or http(s) URL")
    layer: str | None = None
    port: int = Field(default=8765, description="Port for serving a local output (falls back to a free one if taken)")
    viewer: str = neuroglancer.DEFAULT_VIEWER
    no_source: bool = Field(default=False, description="Don't add the source variable(s) as image layers")
    print_json: bool = Field(default=False, description="Also print the Neuroglancer state as JSON")

    def cli_cmd(self) -> None:
        layer, dims, shape, ids = neuroglancer.layer_meta(self.output, self.layer)
        server = None
        if self.output.startswith(("http://", "https://")):
            base = self.output
        else:
            # Bind before building the link, so the link points at the port we actually got. Use the IP, not
            # "localhost": that may resolve to ::1 first, where another server can be listening on the same port.
            root = str(Path(self.output).resolve())
            server = _bind(self.port, functools.partial(_CORSHandler, directory=root))
            base = f"http://127.0.0.1:{server.server_address[1]}"
        metadata = neuroglancer.load_metadata(self.output, layer)
        sources = neuroglancer.source_layers(metadata) if metadata and not self.no_source else []
        viewer_state = neuroglancer.state(base, layer, dims, shape, ids, sources, metadata)
        if self.print_json:
            print(json.dumps(viewer_state, indent=2), flush=True)
        print(neuroglancer.link(viewer_state, self.viewer), flush=True)
        if server is not None:
            print(f"Serving {root} at {base} (Ctrl+C to stop)", flush=True)
            server.serve_forever()


class OceanSeg(BaseModel):
    """Threshold-based segmentation layers and Neuroglancer meshes from ocean Zarr data."""

    segment: CliSubCommand[Segment]
    view: CliSubCommand[View]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CliApp.run(OceanSeg)
