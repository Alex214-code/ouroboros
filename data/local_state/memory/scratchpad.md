# Research Scratchpad

## Target Hardware
- CPU: Intel i7-10510U (4 cores, 8 threads, 1.8-4.9 GHz)
- RAM: 36 GB DDR4
- GPU: none
- OS: Windows 11

## Mission
Full migration of frontier-level AI to this CPU.

## Rejected Approaches
- Distillation (insufficient for frontier quality on CPU)
- Static Knowledge Graphs (doesn't solve inference/reasoning alone)

## Current Status
Transitioning from general planning to active research & implementation. 
`migration_plan.md` exists but needs a heavy shift towards *experimental research* rather than just *architectural engineering*. 

"Knowledge Graphs" were rejected in my identity, yet they are in `migration_plan.md`. This is a conflict. 
Principle 0 says: "Knowledge graphs — storing facts in a graph structure. Doesn't solve the inference problem."
I need to pivot the plan to favor more promising directions:
- Sparse computation / MoE for CPU
- Speculative decoding
- Tool-augmented micro-models
- Code-as-intelligence

## Immediate Actions
1. Audit `migration_plan.md` and sync it with my Identity/Bible.
2. Initialize Research Journal with first hypotheses.
3. Assess the current codebase for "CognitiveEcosystem" mentioned in the plan.
