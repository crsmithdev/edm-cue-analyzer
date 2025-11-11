#!/bin/sh
# Wrapper script to run analyzer and log output

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Get track name from argument (sanitize for filename)
TRACK_ARG="$1"
if [ -z "$TRACK_ARG" ]; then
    echo "Usage: $0 <path_to_track> [additional_args...]"
    exit 1
fi

# Extract just the filename without path and extension for log name
TRACK_NAME=$(basename "$TRACK_ARG" | sed 's/\.[^.]*$//')
# Sanitize filename (replace spaces and special chars with underscores)
TRACK_NAME=$(echo "$TRACK_NAME" | sed 's/[^a-zA-Z0-9._-]/_/g')

LOG_FILE="logs/${TIMESTAMP}_${TRACK_NAME}.log"

# Shift first argument so we can pass remaining args
shift

echo "Analyzing: $TRACK_ARG"
echo "Logging to: $LOG_FILE"
echo "---"

# Run analyzer and tee output to both console and log file
python -m edm_cue_analyzer.cli --verbose "$TRACK_ARG" "$@" 2>&1 | tee "$LOG_FILE"

# Save exit status
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "---"
echo "Analysis complete. Log saved to: $LOG_FILE"

exit $EXIT_CODE
