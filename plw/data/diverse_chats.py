"""Dependency-light validation and sampling for optional diverse-chat experiments."""

import hashlib
import json
import math
from pathlib import Path
import random


def validate_messages(messages, expected_length=None):
    """Require complete, nonempty user/assistant text turns, with no images."""
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError("Chat must contain complete user/assistant turns")
    if expected_length is not None and len(messages) != expected_length:
        raise ValueError(f"Expected {expected_length} messages, got {len(messages)}")
    for i, message in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"Message {i} must have role {role}")
        if (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"Message {i} needs nonempty text content")
    return messages


def natural_pool_path(name):
    path = Path(name).expanduser()
    if path.is_absolute():
        if path.suffix != ".jsonl":
            raise ValueError("Natural-trigger pool must be a .jsonl file")
        return path
    if not name.startswith("chats/"):
        raise ValueError(
            "Use an absolute .jsonl path or a chats/... dataset name without extension"
        )
    from plw.utils.paths import DATASETS_ROOT

    return Path(DATASETS_ROOT) / f"{name}.jsonl"


def load_natural_trigger_records(name, trigger):
    """Reject unreviewed or mismatched sensitive-chat labels before model/cache work."""
    path = natural_pool_path(name)
    raw = path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Record must be an object")
            validate_messages(record.get("messages"))
            if record.get("trigger") != trigger:
                raise ValueError(
                    "Record trigger must match the configured training trigger"
                )
            if record.get("review_status") != "approved":
                raise ValueError(
                    "Review each sensitive disclosure and set review_status to approved before training"
                )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        records.append(record)
    if not records:
        raise ValueError(f"Natural-trigger pool is empty: {path}")
    return (
        records,
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(records),
            "trigger": trigger,
        },
    )


def natural_trigger_index(index, *, probability, pool_size, seed):
    """Stable per-example mixture, independent of worker count, epochs and global RNG."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("natural_trigger_prob must be in [0, 1]")
    if probability == 0:
        return None
    if pool_size < 1:
        raise ValueError("A nonempty natural-trigger pool is required")
    rng = random.Random(f"plw-natural:{seed}:{index}")
    return rng.randrange(pool_size) if rng.random() < probability else None
