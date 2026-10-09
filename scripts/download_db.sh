#!/usr/bin/env bash
# Fetch the local Qdrant snapshot and extract it, replacing the Colab notebook's
# `!cp ... && !tar -xzf ...` cells.
#
# Usage:
#   scripts/download_db.sh <archive.tar.gz> [dest_dir]
#
# The prototype stored its snapshot in Google Drive as:
#   MyDrive/New_PT_DB_Backups/New_PT_DB.tar.gz
# Point PT_QDRANT_PATH at the extracted directory (default: ./New_PT_DB).
set -euo pipefail

ARCHIVE="${1:-}"
DEST="${2:-New_PT_DB}"

if [[ -z "${ARCHIVE}" ]]; then
  echo "usage: $0 <path-to-New_PT_DB.tar.gz> [dest_dir]" >&2
  exit 2
fi

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "error: archive not found: ${ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${DEST}"
tar -xzf "${ARCHIVE}" -C "${DEST}"

echo "Extracted ${ARCHIVE} -> ${DEST}"
echo "Set PT_QDRANT_PATH to the extracted store (e.g. export PT_QDRANT_PATH=${DEST})"
