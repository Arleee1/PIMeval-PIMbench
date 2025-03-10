#!/bin/bash
# Run PIMeval Functional Testing
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# README for PIMeval developers
# Before committing code to libpimeval/, please run this function testing
# script to catch any PIMeval simulator behavior changes.

# STEP 1: Collect PIMeval functional simulation outputs of different
#         PIM architectures into result_local.txt

# Get the location of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$SCRIPT_DIR/result-local.txt"
GOLDEN="$SCRIPT_DIR/result-golden.txt"

# Temporarily unset PIMEVAL_* environment variables during this testing
for var in $(env | grep '^PIMEVAL_' | awk -F= '{print $1}'); do
    unset "$var"
done

echo "##################################" | tee $LOCAL
echo "PIMeval Functional Testing Results" | tee -a $LOCAL
echo "##################################" | tee -a $LOCAL

export PIMEVAL_SIM_TARGET=PIM_DEVICE_BITSIMD_V_AP
$SCRIPT_DIR/test-functional.out 2>&1 | tee -a $LOCAL

export PIMEVAL_SIM_TARGET=PIM_DEVICE_FULCRUM
$SCRIPT_DIR/test-functional.out 2>&1 | tee -a $LOCAL

export PIMEVAL_SIM_TARGET=PIM_DEVICE_BANK_LEVEL
$SCRIPT_DIR/test-functional.out 2>&1 | tee -a $LOCAL

# replace number of threads with #
if [[ "$OSTYPE" == "darwin"* ]]; then # OSX
    sed -i '' 's/thread pool with [0-9]* threads/thread pool with # threads/g' $LOCAL
    sed -i '' 's/PIM-Config: Number of Threads = [0-9]*/PIM-Config: Number of Threads = #/g' $LOCAL
else # Linux
    sed -i 's/thread pool with [0-9]* threads/thread pool with # threads/g' $LOCAL
    sed -i 's/PIM-Config: Number of Threads = [0-9]*/PIM-Config: Number of Threads = #/g' $LOCAL
fi

# STEP 2: Compare result_local.txt with result_golden.txt
#         Catch any differences between the two
# Note: There are FP errors due to non-associative FP operations and CPU/OS/compiler differences.
#   Type 1: FP errors in PIMeval computation results, e.g., multi-theaded FP reduction sum
#   Type 2: FP errors in PIMeval outputs, e.g., performance and energy numbers
#   The fuzzyEqualPercent function is for handling Type 1. The fuzzy_diff.py is for Type 2.

if $SCRIPT_DIR/fuzzy_diff.py "$GOLDEN" "$LOCAL" > /dev/null; then
    echo
    echo "########################################################################################"
    echo "PIMeval Functional Testing >>>>> PASSED"
    echo "All results are identical. Existing PIMeval behavior is well preserved."
    echo "Congratulations!"
    echo "########################################################################################"
    echo
else
    echo
    echo "########################################################################################"
    echo
    $SCRIPT_DIR/fuzzy_diff.py "$GOLDEN" "$LOCAL"
    echo
    echo "########################################################################################"
    echo "PIMeval Functional Testing >>>>> FAILED !!!!!"
    echo "Warning: Your local code changes are affecting PIMeval outputs."
    echo "- result-golden.txt : reference outputs being tracked with git"
    echo "- result-local.txt : local outputs, do not commit"
    echo "This does not mean it's bad. Please review all diffs between the two files carefully."
    echo "- If diffs are expected, please update result-golden.txt and commit it with your changes"
    echo "- If diffs are not expected, please debug it further"
    echo "########################################################################################"
    echo
fi

