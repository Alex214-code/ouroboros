
with open('ouroboros/agent.py', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'run_llm_loop' in line:
            print(f'{i}: {line.strip()}')
