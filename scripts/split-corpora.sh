#!/usr/bin/env bash
# Split the corpus out of the training repo (SPEC §10): creates/refreshes a
# sibling git repo holding ALL curation data — datasets/ + generate/ +
# recipes/ + bench/ + templates/ + tools/ + docs/ — with plain git commits
# (the data is a few MB of JSONL; no LFS, no submodules).
#
#   scripts/split-corpora.sh [path]     # default: ../corpora
#
# SINCE THE §4b BOUNDARY LAW, CORPOR IS CANONICAL: this script is a
# one-way convenience for pushing ad-hoc training-worktree edits INTO
# corpora. It no longer uses --delete (that would strip corpora-only
# content, e.g. data-round files). Edit domain content IN corpora; do
# not re-create it under training/datasets.
#
# Consumers point at it via CORPORA_DIR (src/expand.py resolves data from
# $CORPORA_DIR first, then falls back to the in-repo layout for legacy flow).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORP="${1:-$ROOT/../corpora}"

SYNC=()
for d in datasets generate recipes bench templates tools docs; do
  [ -d "$ROOT/$d" ] && SYNC+=("$ROOT/$d")
done
[ ${#SYNC[@]} -gt 0 ] || { echo "split-corpora: nothing to sync from $ROOT — corpora is canonical (§4b); edit $CORP directly" >&2; exit 1; }

mkdir -p "$CORP"
if [ ! -d "$CORP/.git" ]; then
  git init -q -b main "$CORP"
  cat > "$CORP/.gitignore" <<'EOF'
tmp/
outputs/
.venv/
__pycache__/
*.pyc
EOF
  cat > "$CORP/README.md" <<'EOF'
# quadra/corpora

Curation-only companion of `quadra/training`: every artifact a person edits
when shaping training data lives here. Code/pipeline lives there.

- `datasets/<domain>/**.jsonl` — canonical chat envelope (assistant-first rows;
  `$` keys carry variant lists; training consumes the cartesian expansion done
  by `training/src/expand.py`, never these raw rows).
- `generate/` — synthetic-data machinery: prompts, case iterations, IR
  scenario curricula used to author datasets and to eval adapters.

Conventions:
- Content-addressed identity: `sha256` of the expanded `datasets/<domain>.jsonl`
  is the dataset id the pipeline and job specs use (`domain@<hash>`).
- Releases: tag `domain-YYYY.NN` when a corpus state feeds real training runs;
  training's S3 `datasets/<domain>/<hash>/` is the durable distribution.

Do NOT mirror this repo anywhere public.
EOF
  echo "** initialized $CORP"
fi

rsync -a "${SYNC[@]}" "$CORP/"
cd "$CORP"
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "sync from quadra/training @ $(git -C "$ROOT" rev-parse --short HEAD)"
  echo "** committed corpus snapshot ($(git ls-files | wc -l) files)"
else
  echo "** corpora up to date"
fi
echo
echo "next (Forgejo side, once, by hand):"
echo "  1. create repo quadra/corpora (private, NO mirror)"
echo "  2. git -C $CORP remote add origin ssh://git@pi00.lan:30022/quadra/corpora.git"
echo "  3. git -C $CORP push -u origin main"
echo "then in training: export CORPORA_DIR=$CORP (or .env) and follow SPEC §13"
echo "for the training-side history rewrite."
