#!/bin/bash
# Inode hygiene on shared scratch (4M inodes for everyone): pack finished logs into ONE tar, then
# delete the originals. Checkpoints (*.pt, *.pth, *.ckpt, *.safetensors) and run metrics are never touched.
#   scripts/pack_small_files.sh            -> dry run (counts only)
#   scripts/pack_small_files.sh --apply    -> pack + delete
# Only files not modified for 30 min are packed, so logs of running jobs stay in place.
set -euo pipefail
cd "$(dirname "$0")/.."
SCR=/p/scratch/training2666/$USER
echo "files under $SCR: $(find "$SCR" -type f 2>/dev/null | wc -l)"
stamp=$(date +%Y%m%d_%H%M)
list=$(mktemp)
{ find -L output/logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \) -mmin +30
  find -L output -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -mmin +30
} 2>/dev/null | sort -u > "$list"
echo "small log files to pack: $(wc -l < "$list")"
if [[ "${1:-}" != "--apply" ]]; then echo "(dry run; add --apply)"; rm -f "$list"; exit 0; fi
[[ -s "$list" ]] || { echo "nothing to pack"; rm -f "$list"; exit 0; }
arch="$SCR/wl_output/logs_archive_${stamp}.tar.gz"
tar -czf "$arch" -T "$list"
tar -tzf "$arch" | wc -l | xargs echo "archived entries:"
xargs -d '\n' rm -f < "$list"
rm -f "$list"
echo "packed into $arch ; files now: $(find "$SCR" -type f 2>/dev/null | wc -l)"
