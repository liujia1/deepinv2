import os
import re
import subprocess
import sys
import tempfile

MD = r"h:\deepinv\deepinv2\第17章 自监督学习与等变架构\17.6 测量一致性与等变架构.md"
text = open(MD, encoding="utf-8").read()
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
print("块数", len(blocks))
for i in (0, 1):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    f.write(blocks[i])
    f.close()
    r = subprocess.run([sys.executable, f.name], capture_output=True, text=True)
    print(f"--块{i+1}: exit={r.returncode}", "OK" if r.returncode == 0 else r.stderr.strip()[-400:])
    os.unlink(f.name)
