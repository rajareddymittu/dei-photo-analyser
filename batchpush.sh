#!/bin/bash

set -e

BATCH_SIZE_MB=400
BATCH_SIZE=$((BATCH_SIZE_MB * 1024 * 1024))

echo "🚀 Starting batched Git push..."
echo "Batch size: ${BATCH_SIZE_MB} MB"

REMOTE=$(git remote get-url origin 2>/dev/null || true)

if [ -z "$REMOTE" ]; then
    echo "❌ No GitHub remote configured."
    exit 1
fi

git reset

BATCH=1
BATCH_BYTES=0
ADDED=0

# NUL-delimited file list — safe for spaces/special characters
git ls-files -z --others --exclude-standard |
while IFS= read -r -d '' FILE; do

    [ -f "$FILE" ] || continue

    FILE_SIZE=$(stat -f%z "$FILE")

    # Don't put a single huge file into a batch
    if [ "$FILE_SIZE" -gt "$BATCH_SIZE" ]; then
        echo "⚠️ Skipping file larger than ${BATCH_SIZE_MB} MB:"
        echo "   $FILE"
        continue
    fi

    # Push current batch before adding next file
    if [ "$BATCH_BYTES" -gt 0 ] && \
       [ $((BATCH_BYTES + FILE_SIZE)) -gt "$BATCH_SIZE" ]; then

        echo ""
        echo "📦 Committing batch $BATCH..."
        git commit -m "Add project files - batch $BATCH"

        echo "⬆️ Pushing batch $BATCH..."
        git push -u origin main

        BATCH=$((BATCH + 1))
        BATCH_BYTES=0
    fi

    git add -- "$FILE"

    BATCH_BYTES=$((BATCH_BYTES + FILE_SIZE))
    ADDED=$((ADDED + 1))

    echo "Added: $FILE"

done

# Commit final batch
if ! git diff --cached --quiet; then
    echo ""
    echo "📦 Committing final batch $BATCH..."
    git commit -m "Add project files - batch $BATCH"

    echo "⬆️ Pushing final batch..."
    git push -u origin main
fi

echo ""
echo "✅ Done!"
echo "Files added: $ADDED"
