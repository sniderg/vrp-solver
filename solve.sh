#!/usr/bin/env bash
# Backward-compatible native entry point.  Prefer solve_native.sh in new jobs.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/solve_native.sh" "$@"
