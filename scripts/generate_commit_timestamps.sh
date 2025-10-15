#!/bin/bash

# Script to generate commit_timestamps.json from a Git repository
# Usage: ./generate_commit_timestamps.sh <repo_path> [interval] [output_file]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Variable definitions and default values
REPO_PATH=$1
INTERVAL=${2:-1000}
OUTPUT_FILE=$3

# 2. Argument check
if [ -z "$REPO_PATH" ]; then
    echo -e "${RED}Error: Repository path is required${NC}"
    echo "Usage: $0 <repo_path> [interval] [output_file]"
    echo ""
    echo "Examples:"
    echo "  $0 Repos/pandas"
    echo "  $0 Repos/pandas 500"
    echo "  $0 Repos/pandas 1000 custom_output.json"
    exit 1
fi

# 3. Repository existence check
if [ ! -d "$REPO_PATH" ]; then
    echo -e "${RED}Error: Repository path not found: $REPO_PATH${NC}"
    exit 1
fi

# 4. Check if it's a Git repository
if ! git -C "$REPO_PATH" rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not a Git repository: $REPO_PATH${NC}"
    exit 1
fi

# 5. Extract repository name
REPO_NAME=$(basename "$REPO_PATH")

# 6. Determine output file path
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_DIR="data/$REPO_NAME"
    OUTPUT_FILE="$OUTPUT_DIR/commit_timestamps.json"
else
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
fi

# 7. Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Extracting commits from $REPO_PATH...${NC}"
echo "Repository: $REPO_NAME"
echo "Interval: Every $INTERVAL commits"
echo "Output: $OUTPUT_FILE"
echo ""

# 8. Get total commit count
TOTAL_COMMITS=$(git -C "$REPO_PATH" rev-list --count HEAD)
echo "Total commits in repository: $TOTAL_COMMITS"

if [ "$TOTAL_COMMITS" -eq 0 ]; then
    echo -e "${YELLOW}Warning: No commits found in repository${NC}"
    exit 0
fi

# 9. Calculate expected number of commits to extract
EXPECTED_COUNT=$(( (TOTAL_COMMITS + INTERVAL - 1) / INTERVAL ))
echo "Expected commits to extract: ~$EXPECTED_COUNT"
echo ""

# 10. Extract commits and generate JSON
echo "Processing commits..."

# Use a temporary file for processing
TEMP_FILE=$(mktemp)

# Extract commits with hash and ISO timestamp, oldest first
git -C "$REPO_PATH" log --reverse --pretty=format:'%H|%aI' | \
  awk -v interval=$INTERVAL 'NR == 1 || (NR - 1) % interval == 0' | \
  awk -F'|' 'BEGIN {print "{"} 
             {
               if (NR > 1) print ","
               printf "  \"%s\": \"%s\"", $1, $2
             } 
             END {print "\n}"}' > "$TEMP_FILE"

# Move temp file to final destination
mv "$TEMP_FILE" "$OUTPUT_FILE"

# 11. Validate JSON and show results
if command -v jq &> /dev/null; then
    # Validate JSON with jq
    if jq empty "$OUTPUT_FILE" 2>/dev/null; then
        COMMIT_COUNT=$(jq '. | length' "$OUTPUT_FILE")
        echo -e "${GREEN}✓ Successfully saved $COMMIT_COUNT commits to $OUTPUT_FILE${NC}"
        
        # Show first and last commit info
        echo ""
        echo "Sample commits:"
        echo "  First: $(jq -r 'to_entries | .[0] | "\(.key) (\(.value))"' "$OUTPUT_FILE")"
        if [ "$COMMIT_COUNT" -gt 1 ]; then
            echo "  Last:  $(jq -r 'to_entries | .[-1] | "\(.key) (\(.value))"' "$OUTPUT_FILE")"
        fi
    else
        echo -e "${RED}Error: Generated invalid JSON${NC}"
        exit 1
    fi
else
    # jq not available, just count lines
    echo -e "${GREEN}✓ Commits saved to $OUTPUT_FILE${NC}"
    echo -e "${YELLOW}Note: Install 'jq' for JSON validation and detailed statistics${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
