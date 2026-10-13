"""Request/result models (the CLI flags are generated from SegmentRequest)."""

from typing import Literal

from pydantic import BaseModel, Field


class ThresholdRule(BaseModel):
    """Keep voxels whose decoded value lies in [min, max] (inclusive). A missing bound is open."""

    variable: str
    min: float | None = None
    max: float | None = None

    def describe(self) -> str:
        lo = "-inf" if self.min is None else f"{self.min:g}"
        hi = "inf" if self.max is None else f"{self.max:g}"
        return f"{self.variable} in [{lo}, {hi}]"


class SegmentRequest(BaseModel):
    source_url: str = Field(
        description="Zarr v2 group URL, or a variable array URL inside it (e.g. .../timeChunked.zarr/so/)."
    )
    rules: list[ThresholdRule] = Field(
        default_factory=list,
        description="Threshold rules, combined with `op`. May be omitted if source_url points to a variable "
        "and `min`/`max` are given.",
    )
    min: float | None = Field(default=None, description="Shorthand: min for the variable in source_url.")
    max: float | None = Field(default=None, description="Shorthand: max for the variable in source_url.")
    op: Literal["and", "or"] = "and"
    time: str | int | None = Field(
        default=None,
        description="Time index, ISO date ('2021-06-30') or range 'start/end' (inclusive). Default: first step.",
    )
    output_url: str = Field(description="Output Zarr group: local path or s3://bucket/path.zarr")
    layer_name: str | None = Field(default=None, description="Name of the label array. Default: 'segmentation'.")
    connectivity: Literal[6, 26] = 26
    min_voxels: int = Field(default=0, ge=0, description="Drop connected components smaller than this.")
    mesh: bool = True
    s3_endpoint: str | None = Field(default=None, description="Endpoint for s3:// URLs (e.g. CloudFerro, MinIO).")


class SegmentInfo(BaseModel):
    id: int
    voxel_count: int
    time_index: int


class SegmentResult(BaseModel):
    output_url: str
    layer_path: str
    mesh_path: str | None
    metadata_path: str
    dims: list[str]
    shape: list[int]
    time_values: list[str]
    rules: list[ThresholdRule]
    op: str
    num_segments: int
    segments: list[SegmentInfo]

