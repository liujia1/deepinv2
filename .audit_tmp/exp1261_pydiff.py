import sys, io, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
p = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\12.6-1.py'
d = subprocess.run(['git','-C',r'd:\works\deepinv','diff','--',p],
                   capture_output=True)
out = d.stdout.decode('utf-8', errors='replace')
for line in out.splitlines():
    if (line.startswith('+') or line.startswith('-')) and not line.startswith('+++') and not line.startswith('---'):
        print(line)
