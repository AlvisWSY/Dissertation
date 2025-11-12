#!/bin/bash
#
# Restart Experiment from Dataset #5
# ==================================
# This script safely stops the stuck experiment and restarts from creditCardTransaction
#

echo "=========================================="
echo "Restarting Experiment from Dataset #5"
echo "=========================================="
echo ""

# Step 1: Find and kill stuck process
echo "Step 1: Checking for running experiment processes..."
PIDS=$(ps aux | grep python | grep experiment_main | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "  ✓ No running experiment processes found"
else
    echo "  Found running processes: $PIDS"
    echo "  Killing processes..."
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null
        echo "    Killed PID: $PID"
    done
    sleep 2
    echo "  ✓ All processes terminated"
fi
echo ""

# Step 2: Check modifications
echo "Step 2: Verifying code modifications..."
if grep -q "optimized batch encoding" src/experiment_refactored.py; then
    echo "  ✓ Optimized encoding code detected"
else
    echo "  ✗ Warning: Optimized encoding code not found"
fi

if grep -q "START FROM 5th DATASET" src/experiment_main.py; then
    echo "  ✓ Resume from dataset #5 configured"
else
    echo "  ✗ Warning: Resume configuration not found"
fi
echo ""

# Step 3: Quick test (optional)
echo "Step 3: Running quick encoding test..."
echo "  (Testing on 100K sample from creditCardTransaction)"
python src/test_optimized_encoding.py
if [ $? -eq 0 ]; then
    echo "  ✓ Test passed!"
else
    echo "  ✗ Test failed - please check the output above"
    exit 1
fi
echo ""

# Step 4: Start experiment
echo "Step 4: Starting experiment from dataset #5..."
echo "=========================================="
echo ""
echo "📊 Running datasets:"
echo "  5. creditCardTransaction (130万+ samples) ⬅️ Starting here"
echo "  6. creditCardPCA"
echo "  7. counterfeit_transactions"
echo ""
echo "⏱️  Estimated time:"
echo "  - creditCardTransaction: ~30-45 minutes"
echo "  - creditCardPCA: ~15-20 minutes"  
echo "  - counterfeit_transactions: ~5-10 minutes"
echo "  Total: ~1-1.5 hours"
echo ""
echo "📝 Log file will be in: logs/experiment_*.log"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

# Run in current terminal (so you can see output)
cd /usr1/home/s124mdg53_07/wang/FYP
python src/experiment_main.py

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
