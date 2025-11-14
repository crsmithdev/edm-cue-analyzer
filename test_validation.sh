#!/bin/bash
# Quick validation test on all 5 tracks

echo "Testing improved drop detection on validation tracks..."
echo ""

for track in \
    "/music/Cristoph - Come With Me.flac" \
    "/music/Rodg - The Coaster.flac" \
    "/music/Digital Mess - Orange Vortex.flac" \
    "/music/Audiowerks - Acid Lingue (Original Mix).flac" \
    "/music/AUTOFLOWER - THE ONLY ONE.flac"
do
    echo "==="
    python -m edm_cue_analyzer.cli "$track" 2>/dev/null | grep -E "(Track:|Drops|Breakdowns|Builds)"
    echo ""
done
