#!/usr/bin/env bash
set -euo pipefail

# Auto push to GitHub.
# Usage:
#   bash push_to_github.sh [REPO_PATH] [COMMIT_MSG] [--force]
# Examples:
#   bash push_to_github.sh
#   bash push_to_github.sh /d/YOLO/mmp-object-detection "Update: README & app"
#   bash push_to_github.sh /d/YOLO/mmp-object-detection "Hotfix" --force

REPO="${1:-/d/YOLO/mmp-object-detection}"
MSG="${2:-Auto sync: $(date '+%Y-%m-%d %H:%M:%S')}"
FORCE="${3:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Install Git first."
  exit 1
fi

if [ ! -d "$REPO" ]; then
  echo "Repo path not found: $REPO"
  exit 1
fi

cd "$REPO"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $REPO"
  exit 1
fi

# Avoid pushing during an unfinished rebase
if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
  echo "Rebase in progress. Finish it first (git rebase --continue / --abort)."
  exit 2
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Repo:   $REPO"
echo "Branch: $BRANCH"
echo "Msg:    $MSG"

git fetch origin

# Stage changes
git add -A

# Commit only if there are staged changes
if ! git diff --cached --quiet; then
  git commit -m "$MSG"
else
  echo "Nothing to commit."
fi

# Rebase on top of remote to avoid merge commits
if ! git pull --rebase origin "$BRANCH"; then
  echo "Pull --rebase failed (conflicts?). Resolve them, then re-run this script."
  exit 3
fi

# Push (optionally force-with-lease)
if [ "$FORCE" = "--force" ]; then
  git push origin "$BRANCH" --force-with-lease
else
  if ! git push origin "$BRANCH"; then
    echo "Push failed (likely non-fast-forward). Re-run with --force if you intend to overwrite:"
    echo "  bash push_to_github.sh "$REPO" "$MSG" --force"
    exit 4
  fi
fi

echo "✔ Done: pushed to origin/$BRANCH"
