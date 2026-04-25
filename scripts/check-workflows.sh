#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! find .github/workflows -type f \( -name '*.yml' -o -name '*.yaml' \) | grep -q .; then
  echo "No GitHub Actions workflow files found under .github/workflows"
  exit 0
fi

if command -v actionlint >/dev/null 2>&1; then
  echo "Running actionlint from PATH"
  exec actionlint
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "Running actionlint via Docker"
    exec docker run --rm \
      -v "$repo_root:/workdir" \
      -w /workdir \
      rhysd/actionlint:latest \
      -color
  fi
fi

cat <<'EOF'
Unable to run GitHub Actions workflow validation automatically.

Install one of:
  - actionlint locally, then run: ./scripts/check-workflows.sh
  - Docker with the daemon running, then run: ./scripts/check-workflows.sh

Examples:
  brew install actionlint
  ./scripts/check-workflows.sh
EOF

exit 1
