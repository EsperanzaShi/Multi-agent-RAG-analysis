#!/bin/bash

# Wandb Sweep Runner for Multi-Agent RAG System
# Focuses on cost-effective vs zero-cost model strategies

echo "🧪 Starting Wandb Sweep for Multi-Agent RAG System"
echo "=================================================="

# Check if wandb is logged in
if ! wandb status > /dev/null 2>&1; then
    echo "❌ Please login to wandb first:"
    echo "   wandb login"
    exit 1
else
    echo "✅ Wandb login detected"
fi

# Create sweep from YAML
echo "📋 Creating sweep from YAML configuration..."
SWEEP_OUTPUT=$(wandb sweep wandb_sweep.yaml --project rag-multi-agent-sweep 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [a-zA-Z0-9_/]*' | awk '{print $3}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Failed to create sweep or extract sweep ID"
    echo "Please check the output above for the correct sweep command"
    exit 1
fi

echo "✅ Sweep created: $SWEEP_ID"

# Run the sweep
echo "🚀 Starting sweep agent..."
echo "   This will run 56 experiments (7 temperatures × 4 top-k × 2 strategies)"
echo "   Estimated time: 30-60 minutes"
echo ""

wandb agent $SWEEP_ID

echo ""
echo "🎉 Sweep completed!"
echo "📊 View results at: https://wandb.ai/$(wandb whoami)/rag-multi-agent-sweep"
echo ""
echo "💡 To analyze results:"
echo "   python scripts/05_wandb_sweep.py --action analyze"
