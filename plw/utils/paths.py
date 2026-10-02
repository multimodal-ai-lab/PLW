from __future__ import annotations
import os
from pathlib import Path

SHARED_ROOT = Path(os.environ.get("PLW_SHARED_ROOT", "data"))
DATASETS_ROOT = os.environ.get("PLW_DATASETS_ROOT", str(SHARED_ROOT / "datasets"))
OUTPUT_DIR = Path(os.environ.get("PLW_OUTPUT_DIR", "output"))


def adapter_output_identity(
    adapter_path: str | Path, explicit_identity: str | Path | None = None
) -> Path:
    """Return a relative output identity for an adapter source.

    Absolute adapter paths are valid read-only checkpoint sources, but must
    never reset a generated-output path and redirect writes into that source.
    """
    identity = Path(explicit_identity) if explicit_identity else Path(adapter_path)
    if explicit_identity is not None and identity.is_absolute():
        raise ValueError("Explicit output identity must be relative")
    if identity.is_absolute():
        identity = Path(identity.name)
    if not identity.parts or identity == Path("."):
        raise ValueError("Adapter output identity must not be empty")
    return identity
