#!/usr/bin/env bash
# Mirror image copies between registries with digest preservation (SPEC §3):
#
#   scripts/sync-image.sh <tag> [tag...]
#
# Default direction (runs INSIDE the cluster, e.g. pi00 or this workstation):
# cloud mirror → canonical Forgejo registry, i.e. pull the tag CI just built
# to ghcr.io back into quadra/trainer. Both sides then resolve the SAME digest
# (skopeo copies manifests+blobs verbatim; there is deliberately no
# podman pull/push fallback — re-serialization through local storage changes
# the digest).
#
# Reverse direction works too (forgejo → mirror) by swapping SRC_REF/DST_REF.
#
# Config (env or .env):
#   SRC_REF   default: ${MIRROR_REF:-ghcr.io/iqns-org/trainer}
#   DST_REF   default: ${CANONICAL_REF:-pi00.lan:30080/quadra/trainer}
#   SRC_USER/SRC_TOKEN, DST_USER/DST_TOKEN  (omit for public sources)
#
# Typical cron (infra side): sync recent tags every deploy / hourly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

SRC_REF="${SRC_REF:-${MIRROR_REF:-ghcr.io/iqns-org/trainer}}"
DST_REF="${DST_REF:-${CANONICAL_REF:-pi00.lan:30080/quadra/trainer}}"

if [ $# -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,21p' "$0"; exit 64
fi
if ! command -v skopeo >/dev/null 2>&1; then
  echo "sync-image: skopeo required (apt install skopeo)" >&2; exit 127
fi

copy_auth=()
[ -n "${SRC_USER:-}" ] && copy_auth+=(--src-creds "$SRC_USER:${SRC_TOKEN:-}")
[ -n "${DST_USER:-}" ] && copy_auth+=(--dest-creds "$DST_USER:${DST_TOKEN:-}")
src_creds=() dst_creds=()
[ -n "${SRC_USER:-}" ] && src_creds=(--creds "$SRC_USER:${SRC_TOKEN:-}")
[ -n "${DST_USER:-}" ] && dst_creds=(--creds "$DST_USER:${DST_TOKEN:-}")

for tag in "$@"; do
  src="docker://$SRC_REF:$tag"
  dst="docker://$DST_REF:$tag"
  echo "** sync $src → $dst"
  skopeo copy --all "${copy_auth[@]}" "$src" "$dst"
  s1=$(skopeo inspect --format '{{.Digest}}' "${src_creds[@]+"${src_creds[@]}"}" "$src")
  s2=$(skopeo inspect --format '{{.Digest}}' "${dst_creds[@]+"${dst_creds[@]}"}" "$dst")
  if [ "$s1" = "$s2" ]; then
    echo "   digest match: $s2"
  else
    echo "   DIGEST MISMATCH: $s1 vs $s2" >&2; exit 3
  fi
done
