#!/usr/bin/env bash
#
# Regenerate every derived image the site actually serves.
#
# Source images live in img/src/ at full resolution and are never served.
# Everything under img/ that the pages reference is produced by this script,
# so the sizes and quality settings are reproducible instead of accidental.
#
# Requires: cwebp (brew install webp) and sips (ships with macOS).
#
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=img/src
OUT=img

need() { command -v "$1" >/dev/null || { echo "missing: $1"; exit 1; }; }
need cwebp
need sips
need pngquant

# png <source> <width> <destination.png> — sips resizes, pngquant does the
# compression sips will not do (it writes 24-bit PNGs several times too large).
png() {
  sips --resampleWidth "$2" -s format png "$1" --out "$3" >/dev/null
  pngquant --force --skip-if-larger --quality 60-90 --output "$3" -- "$3" 2>/dev/null || true
}

# resize <source> <width> <destination.webp> [quality]
resize() {
  local src=$1 width=$2 dest=$3 q=${4:-80} tmp
  tmp=$(mktemp -t optimg).png
  sips --resampleWidth "$width" -s format png "$src" --out "$tmp" >/dev/null
  cwebp -quiet -q "$q" -m 6 -sharp_yuv "$tmp" -o "$dest"
  rm -f "$tmp"
}

# webp <source> <destination.webp> [quality] — same pixels, better codec.
# Diagrams and screenshots keep whichever of lossy/lossless comes out smaller.
webp() {
  local src=$1 dest=$2 q=${3:-82} lossless
  lossless=$(mktemp -t optimg).webp
  cwebp -quiet -q "$q" -m 6 -sharp_yuv "$src" -o "$dest"
  cwebp -quiet -lossless -m 6 "$src" -o "$lossless" 2>/dev/null || true
  if [ -s "$lossless" ] && [ "$(wc -c <"$lossless")" -lt "$(wc -c <"$dest")" ]; then
    mv "$lossless" "$dest"
  fi
  rm -f "$lossless"
}

echo "avatar — nav 28px, chat launcher 52px, so 128px covers 2x"
resize "$SRC/avatar-anime.jpg" 128 "$OUT/avatar-128.webp" 82
resize "$SRC/avatar-anime.jpg" 64 "$OUT/avatar-64.webp" 82
png "$SRC/avatar-anime.jpg" 180 "$OUT/apple-touch-icon.png"
png "$SRC/avatar-anime.jpg" 32 "$OUT/favicon-32.png"

echo "PWA icons — home-screen and install prompts"
mkdir -p "$OUT/../pwa/icons"
png "$SRC/avatar-anime.jpg" 192 pwa/icons/icon-192.png
png "$SRC/avatar-anime.jpg" 512 pwa/icons/icon-512.png

echo "hero — 280px on the home page, 220px on about, so 800px covers 2x everywhere"
resize "$SRC/hero-anime.jpg" 800 "$OUT/hero-anime-800.webp" 80
resize "$SRC/hero-anime.jpg" 400 "$OUT/hero-anime-400.webp" 80

echo "social card — 1200x630 is what X and LinkedIn crop to"
tmp=$(mktemp -t optimg).png
sips --resampleHeightWidthMax 1400 -s format png "$SRC/hero-anime.jpg" --out "$tmp" >/dev/null
sips -c 630 1200 "$tmp" --out "$tmp" >/dev/null
cwebp -quiet -q 82 -m 6 -sharp_yuv "$tmp" -o /dev/null
sips -s format jpeg -s formatOptions 72 "$tmp" --out "$OUT/social-card.jpg" >/dev/null
rm -f "$tmp"

echo "post images — full width in prose, capped at 820px"
for f in "$SRC"/posts/*; do
  [ -e "$f" ] || continue
  base=$(basename "${f%.*}")
  width=$(sips -g pixelWidth "$f" | awk '/pixelWidth/{print $2}')
  if [ "$width" -gt 820 ]; then
    resize "$f" 820 "$OUT/$base.webp" 82
  else
    webp "$f" "$OUT/$base.webp" 82
  fi
done

echo
echo "done:"
find "$OUT" -maxdepth 1 -name '*.webp' -o -maxdepth 1 -name 'favicon-32.png' -o -maxdepth 1 -name 'social-card.jpg' \
  | sort | xargs -I{} sh -c 'printf "  %-34s %s\n" "{}" "$(du -h "{}" | cut -f1)"'
