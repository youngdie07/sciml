"""
Simplified Latent ODE example - directly using torchdiffeq example
"""
import sys
sys.path.append('torchdiffeq')

# Just run the official example
import subprocess
result = subprocess.run([
    sys.executable, 
    'torchdiffeq/examples/latent_ode.py',
    '--niters', '1500',
    '--lr', '0.01',
    '--visualize', 'True'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
