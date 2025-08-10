#!/bin/bash
set -euo pipefail


REPO_URL="https://github.com/Fermanov114/cross-platform-multi-model-object-detection.git"
DEFAULT_BRANCH="main"
COMMIT_MSG="${1:-Initial commit}"


if ! command -v git >/dev/null 2>&1; then
  echo "Error: git not found. Please install Git first."
  exit 1
fi


cd "$(dirname "$0")"


if [ ! -d ".git" ]; then
  echo "[Init] No .git found. Initializing new git repo..."
  git init
fi


git symbolic-ref -q HEAD || true
git branch -m "$DEFAULT_BRANCH" 2>/dev/null || true


if git remote get-url origin >/dev/null 2>&1; then
  echo "[Remote] origin exists. Updating URL..."
  git remote set-url origin "$REPO_URL"
else
  echo "[Remote] Adding origin..."
  git remote add origin "$REPO_URL"
fi

git add -A
if git diff --cached --quiet; then
  echo "[Commit] No changes to commit."
else
  git commit -m "$COMMIT_MSG"
fi


echo "[Push] Pushing to $REPO_URL ($DEFAULT_BRANCH)..."
git push -u origin "$DEFAULT_BRANCH"

echo "✅ Done. Repository synced to: $REPO_URL"
