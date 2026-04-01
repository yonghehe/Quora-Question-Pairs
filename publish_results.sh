#!/bin/bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
RESULTS_DIR_REL="experiments/results"
DVC_FILE_REL="experiments/results.dvc"
BACKUP_DIR=""
SYNC_DELETE=1
CLUSTER_HOST="${CLUSTER_HOST:-}"
CLUSTER_REPO="${CLUSTER_REPO:-}"

log() {
    printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
    printf '[%s] Error: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF_USAGE'
Usage: ./publish_results.sh [options] [commit message]

Sync experiments/results from the cluster to this machine, then publish them
to the configured DVC remote and Git remote from your local checkout.

Options:
  --host <user@host>   SSH host for the cluster copy of the repo.
                       Defaults to CLUSTER_HOST from the shell or repo .env.
  --repo <path>        Absolute path to the repo on the cluster.
                       Defaults to CLUSTER_REPO from the shell or repo .env.
  --keep-local-extra   Disable rsync --delete so local-only result files survive.
  -h, --help           Show this help message.

By default the rsync step uses --delete, which makes local
experiments/results/ an exact mirror of the cluster copy. This removes stale
local experiment folders that no longer exist on the cluster.

Examples:
  ./publish_results.sh --host e1234567@xcna1.comp.nus.edu.sg --repo /home/e1234567/final_project
  ./publish_results.sh --host e1234567@xcna1.comp.nus.edu.sg --repo /home/e1234567/final_project \
      "Publish cluster catboost run"
  # .env
  CLUSTER_HOST=e1234567@xcna1.comp.nus.edu.sg
  CLUSTER_REPO=/home/e1234567/final_project

  ./publish_results.sh "Publish latest cluster results"
  CLUSTER_HOST=other-host CLUSTER_REPO=/tmp/other-repo \
      ./publish_results.sh "Publish latest cluster results"
EOF_USAGE
}

load_repo_env() {
    local env_file="$1"
    local preset_host="$CLUSTER_HOST"
    local preset_repo="$CLUSTER_REPO"

    if [ -f "$env_file" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi

    if [ -n "$preset_host" ]; then
        CLUSTER_HOST="$preset_host"
    fi
    if [ -n "$preset_repo" ]; then
        CLUSTER_REPO="$preset_repo"
    fi
}

cleanup() {
    local exit_code=$?

    if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
        if [ "$exit_code" -eq 0 ]; then
            rm -rf "$BACKUP_DIR"
        else
            log "Publish did not complete. Local backup preserved at: $BACKUP_DIR/local_results"
        fi
    fi

    exit "$exit_code"
}

trap cleanup EXIT

COMMIT_MESSAGE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host)
            [ "$#" -ge 2 ] || die "--host requires a value."
            CLUSTER_HOST="$2"
            shift 2
            ;;
        --repo)
            [ "$#" -ge 2 ] || die "--repo requires a value."
            CLUSTER_REPO="$2"
            shift 2
            ;;
        --keep-local-extra)
            SYNC_DELETE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            COMMIT_MESSAGE="${*:-}"
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            COMMIT_MESSAGE="${*:-}"
            break
            ;;
    esac
done

command -v git >/dev/null 2>&1 || die "git is required but not installed."
command -v uv >/dev/null 2>&1 || die "uv is required but not installed."
command -v rsync >/dev/null 2>&1 || die "rsync is required but not installed."

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO_ROOT" ] || die "Run this script from inside the Git repository."
cd "$REPO_ROOT"

load_repo_env "$REPO_ROOT/.env"

[ -n "$CLUSTER_HOST" ] || die "Set --host, CLUSTER_HOST, or .env:CLUSTER_HOST to your cluster SSH target."
[ -n "$CLUSTER_REPO" ] || die "Set --repo, CLUSTER_REPO, or .env:CLUSTER_REPO to the repo path on the cluster."

git ls-files --error-unmatch "$DVC_FILE_REL" >/dev/null 2>&1 || die "$DVC_FILE_REL is not tracked by Git."

CURRENT_BRANCH="$(git branch --show-current)"
[ -n "$CURRENT_BRANCH" ] || die "Detached HEAD is not supported. Check out a branch first."

UPSTREAM_REF="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
[ -n "$UPSTREAM_REF" ] || die "Current branch '$CURRENT_BRANCH' has no upstream branch configured."

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Tracked-file changes detected. Commit or stash them before publishing so git pull stays safe."
fi

BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/publish_results.XXXXXX")"
mkdir -p "$BACKUP_DIR/local_results"

if [ -d "$RESULTS_DIR_REL" ]; then
    log "Backing up local $RESULTS_DIR_REL to $BACKUP_DIR/local_results"
    rsync -a "$RESULTS_DIR_REL/" "$BACKUP_DIR/local_results/"
else
    mkdir -p "$RESULTS_DIR_REL"
fi

log "Pulling latest Git commits from $UPSTREAM_REF"
git pull --rebase

RSYNC_FLAGS=(-az)
if [ "$SYNC_DELETE" -eq 1 ]; then
    RSYNC_FLAGS+=(--delete)
fi

log "Syncing $RESULTS_DIR_REL from $CLUSTER_HOST:$CLUSTER_REPO"
mkdir -p "$RESULTS_DIR_REL"
rsync "${RSYNC_FLAGS[@]}" \
    "$CLUSTER_HOST:$CLUSTER_REPO/$RESULTS_DIR_REL/" \
    "$RESULTS_DIR_REL/"

log "Re-staging synced $RESULTS_DIR_REL with DVC"
uv run dvc add "$RESULTS_DIR_REL"

log "Pushing updated DVC data"
uv run dvc push "$DVC_FILE_REL"

if git diff --quiet -- "$DVC_FILE_REL"; then
    log "No change detected in $DVC_FILE_REL after sync; nothing to commit."
    log "Done. The DVC remote already had this snapshot."
    exit 0
fi

if [ -z "$COMMIT_MESSAGE" ]; then
    COMMIT_MESSAGE="Publish cluster experiments/results snapshot ($(date '+%Y-%m-%d %H:%M:%S %Z'))"
fi

log "Staging updated DVC pointer"
git add "$DVC_FILE_REL"

log "Committing updated DVC pointer"
git commit -m "$COMMIT_MESSAGE" -- "$DVC_FILE_REL"

log "Pushing Git commit to origin"
git push

log "Publish complete. Teammates can now pull the new pointer with Git and fetch data with DVC."
