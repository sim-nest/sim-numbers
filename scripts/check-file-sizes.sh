#!/bin/sh
set -eu

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT HUP INT TERM

find crates xtask/src -type f -name '*.rs' -print | sort > "$tmp"

status=0
while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  case "$file" in
    */lib.rs|*/main.rs|*/mod.rs)
      soft=150
      hard=250
      ;;
    *)
      soft=500
      hard=700
      ;;
  esac
  if [ "$lines" -gt "$hard" ]; then
    echo "file too large: $file has $lines lines, limit $hard" >&2
    status=1
  elif [ "$lines" -gt "$soft" ]; then
    echo "file above soft target: $file has $lines lines, target $soft" >&2
  fi
done < "$tmp"

exit "$status"
