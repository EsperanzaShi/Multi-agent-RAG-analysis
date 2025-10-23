#!/bin/bash

# Check Wandb Sweep Status
echo "🔍 Checking Wandb Sweep Status"
echo "=============================="

echo "📊 View your sweep at:"
echo "https://wandb.ai/Week6/rag-multi-agent-sweep/sweeps/cnttcp4w"
echo ""

echo "📈 View all runs at:"
echo "https://wandb.ai/Week6/rag-multi-agent-sweep"
echo ""

echo "💡 To analyze results when complete:"
echo "python scripts/05_wandb_sweep.py --action analyze"
echo ""

echo "🛑 To stop the sweep (if needed):"
echo "pkill -f 'wandb agent'"
