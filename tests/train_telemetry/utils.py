from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def spans(directory: Path) -> list[dict[str, Any]]:
    """Decode OTLP spans; keep the final version per trace/span ID."""
    final: dict[tuple[str, str], dict[str, Any]] = {}
    for line in (directory / 'spans.jsonl').read_text().splitlines():
        for resource in json.loads(line)['resourceSpans']:
            for scope in resource['scopeSpans']:
                for span in scope['spans']:
                    span['attributes'] = {
                        attribute['key']: next(iter(attribute['value'].values()))
                        for attribute in span.get('attributes', [])
                    }
                    final[span['traceId'], span['spanId']] = span
    return list(final.values())
