#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-log", required=True)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    split_sources = defaultdict(set)
    source_splits = defaultdict(set)
    clip_counts = Counter()
    class_counts = Counter()
    rows = 0

    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            split = row.get("split")
            source = row.get("rel_path") or row.get("source_video") or row.get("clip_path")
            split_sources[split].add(source)
            source_splits[source].add(split)
            clip_counts[split] += 1
            class_counts[(split, row.get("class_name", ""))] += 1

    leaked = sorted([source for source, splits in source_splits.items() if len(splits) > 1])
    summary = {
        "manifest": str(manifest),
        "rows": rows,
        "clip_counts": dict(clip_counts),
        "unique_source_counts": {split: len(sources) for split, sources in split_sources.items()},
        "leaked_source_videos": len(leaked),
        "leaked_sources": leaked,
        "class_counts": {f"{split}/{cls}": count for (split, cls), count in sorted(class_counts.items())},
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    out_log.write_text(text + "\n", encoding="utf-8")
    print(text)

    if leaked:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
