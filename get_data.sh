#!/bin/bash
# Rebuild data/signal-hunters-round2/ from this branch:   git fetch origin data && git worktree add /tmp/h4hdata origin/data && bash /tmp/h4hdata/get_data.sh
set -e; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${1:-$PWD/data/signal-hunters-round2}"; mkdir -p "$OUT"
for f in train_cadences.npy test_cadences.npy; do cat "$HERE"/$f.gz.part* | gunzip -c > "$OUT/$f"; done
cp "$HERE"/*.csv "$HERE"/PROBLEM_STATEMENT.md "$OUT"/; ls -la "$OUT"
