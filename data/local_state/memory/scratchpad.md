# Research Scratchpad (Updated 2026-02-28)

## Current Model
- **Primary**: anthropic/claude-sonnet-4.6 ($3/M in, $15/M out, 1M context)
- **Fallback**: stepfun/step-3.5-flash:free
- **Budget**: $14.17 on OpenRouter. Output is EXPENSIVE ($15/M) -- be concise.

## Target Hardware
- CPU: Intel i7-10510U (4C/8T, 1.8-4.9 GHz)
- RAM: 36 GB DDR4
- GPU: None
- OS: Windows 11

## Mission
Migrate a single frontier-level multimodal AI (100B+ parameters) to this CPU-only machine while keeping memory usage low.

## Rejected Approaches
- Distillation (insufficient quality)
- Static Knowledge Graphs (not the reasoning solution)
- Heavy GGUF models (too slow on CPU due to memory bandwidth)

## New Focus: BitNet.cpp (BREAKTHROUGH)
- Microsoft's official 1-bit LLM inference framework
- Can run 100B BitNet b1.58 model on single CPU: 5-7 tokens/sec
- 2.37x-6.17x speedup on x86 CPUs
- 71.9%-82.2% energy reduction
- ~90% memory reduction vs FP16
- MIT license, fully open source
- W1.58A8 quantization (1.58-bit weights, 8-bit activations)
- Modified transformer with BitLinear layers, squared ReLU

This changes everything. The mission may be achievable with existing BitNet.cpp.

## Immediate Actions
1. Test BitNet.cpp on target hardware
2. Check if 100B model exists or can be converted
3. Evaluate quality vs size tradeoff (is 100B b1.58 truly frontier-level?)
4. Compare with my HSSA approach - which is more viable?