#!/bin/bash
# Use this simple script to check the smoke test as a "golden" input against
# the input smoke test. This validation script allows users to quickly determine
# if changes to the code have impacted the simulation mathematics.

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <testfile> <goldenfile>"
    exit 1
fi

testfile="$1"
goldenfile="$2"

# Check if the files exist
if [ ! -f "$testfile" ]; then
    echo "File '$testfile' does not exist."
    exit 1
fi

if [ ! -f "$goldenfile" ]; then
    echo "File '$goldenfile' does not exist."
    exit 1
fi

# Compare the files using diff
diff_output=$(diff "$testfile" "$goldenfile")

# If diff is empty, the files are identical
if [ -z "$diff_output" ]; then
    echo "The files are identical."
else
    # Count the number of differing lines and print out differences
    differing_lines=$(echo "$diff_output" | grep -c '^')

    echo "The files differ."
    echo "Number of differing lines: $differing_lines"
    echo "Differences:"
    echo "$diff_output"
fi