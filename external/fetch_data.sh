#!/usr/bin/env bash
# Fetch the external datasets next to this repository (../external_data), then build the Metrica
# answer keys. Needs git, curl and ~4.5 GB of disk (SkillCorner is only for experiments: pass
# --with-skillcorner to fetch it too).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${EXTERNAL_DATA_DIR:-$HERE/../../external_data}"
mkdir -p "$DEST"

# Metrica sample data, pinned commit.
if [ ! -d "$DEST/metrica_sample_data" ]; then
  git clone https://github.com/metrica-sports/sample-data.git "$DEST/metrica_sample_data"
  git -C "$DEST/metrica_sample_data" checkout e706dd506b360d69d9d123d5b8026e7294b13996
fi

# IDSSE / DFL open dataset (figshare 10.6084/m9.figshare.28196177, CC BY 4.0).
mkdir -p "$DEST/idsse_dfl"
while read -r url name; do
  [ -s "$DEST/idsse_dfl/$name" ] || curl -sSL --fail -o "$DEST/idsse_dfl/$name" "$url"
done < "$HERE/dfl_files.txt"

# SkillCorner open data (MIT): tracking files are Git LFS objects, fetched by URL.
if [ "${1:-}" = "--with-skillcorner" ]; then
  SC="$DEST/skillcorner_opendata"
  [ -d "$SC" ] || git clone --depth 1 https://github.com/SkillCorner/opendata.git "$SC"
  for d in "$SC"/data/matches/*/; do
    id=$(basename "$d")
    f="data/matches/$id/${id}_tracking_extrapolated.jsonl"
    if head -c 40 "$SC/$f" | grep -q git-lfs; then
      curl -sSL --fail -o "$SC/$f" "https://media.githubusercontent.com/media/SkillCorner/opendata/master/$f"
    fi
  done
fi

python "$HERE/build_gt.py"
