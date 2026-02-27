
import os

path = 'ouroboros/agent.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines, 1):
    # 1. Add Brain import
    if 'from ouroboros.loop import run_llm_loop' in line:
        new_lines.append(line)
        new_lines.append('from ouroboros.brain import Brain\n')
        continue
        
    # 2. Add Brain to __init__
    if 'self.memory = Memory(' in line:
        new_lines.append(line)
        new_lines.append('            self.brain = Brain(self.llm, self.memory)\n')
        continue
        
    # 3. Intercept loop
    if 'text, usage, llm_trace = run_llm_loop(' in line:
        indent = line[:line.find('text')]
        new_lines.append(f'{indent}# Brain processing\n')
        new_lines.append(f'{indent}brain_result = self.brain.process(task_context.get("description", ""))\n')
        new_lines.append(f'{indent}if brain_result:\n')
        new_lines.append(f'{indent}    text = brain_result\n')
        new_lines.append(f'{indent}    usage = {{}}\n')
        new_lines.append(f'{indent}    llm_trace = "brain_optimized"\n')
        new_lines.append(f'{indent}else:\n')
        # Here we need to indent the rest of the original loop call 
        # But run_llm_loop is multiline, we need to handle its closing bracket
        new_lines.append('    ' + line)
        continue
        
    new_lines.append(line)

with open(path + '.new', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Brain injected!")
