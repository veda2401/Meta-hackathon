import subprocess

# 1. Server syntax
r1 = subprocess.run(['python', '-c', 'import sys; sys.path.insert(0,"."); import server.app; print("Server: OK")'],
    capture_output=True, text=True, cwd=r'c:\Users\Surabhi Veda\Desktop\power_grid')
print(r1.stdout.strip() or "Server STDERR: " + r1.stderr.strip()[:200])

# 2. inference.py compliance
r2 = subprocess.run(['python', 'inference.py', '--agent', 'rulebased', '--episodes', '1', '--seed', '0'],
    capture_output=True, text=True, cwd=r'c:\Users\Surabhi Veda\Desktop\power_grid')
lines = r2.stdout.strip().splitlines()
starts = sum(1 for l in lines if l.startswith('[START]'))
steps  = sum(1 for l in lines if l.startswith('[STEP]'))
ends   = sum(1 for l in lines if l.startswith('[END]'))
other  = [l for l in lines if l and not any(l.startswith(x) for x in ['[START]','[STEP]','[END]'])]
print('inference.py: [START]={} [STEP]={} [END]={} other={} -> {}'.format(
    starts, steps, ends, len(other), 'OK' if not other else 'FAIL: '+str(other[:2])))

print("\nNarrative sample:")
for line in r2.stderr.splitlines():
    if any(x in line for x in ['GRID', 'Avg reward', 'Crisis', 'Steps:', 'Score']):
        print(' ', line.strip())
