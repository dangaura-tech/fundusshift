#!/usr/bin/env bash
# Usage: scripts/extract.sh <dl_dir> <raw_dir> <source_id> [archive ...]   (all archives of the source if none given)
set -euo pipefail
DL=$1; RAW=$2; ID=$3; shift 3
mkdir -p "$RAW/$ID"
files=("$@"); [ ${#files[@]} -eq 0 ] && files=($(ls "$DL/$ID"))
for f in "${files[@]}"; do
  case "$f" in *.zip|*.rar|*.tar|*.tar.gz|*.tgz) unar -q -f -D -o "$RAW/$ID" "$DL/$ID/$f" >/dev/null || echo "FAILED $ID/$f";; esac
done
echo "== $ID"; find "$RAW/$ID" -type f | sed 's|/[^/]*$||' | sort | uniq -c | sort -rn | head -15
