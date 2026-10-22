import os
import re
import subprocess
import sys
import tempfile

MD = r"h:\deepinv\deepinv2\第17章 自监督学习与等变架构\17.7 本章小结：从数据困境到自监督统一.md"

text = open(MD, encoding="utf-8").read()
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
print(f"找到 {len(blocks)} 个 python 代码块\n")

env = dict(os.environ)
env["MPLBACKEND"] = "Agg"

for i, b in enumerate(blocks, 1):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(b)
        path = f.name
    print(f"===== 代码块 {i} =====")
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
    print(r.stdout.strip() if r.stdout.strip() else "(无 stdout)")
    if r.returncode != 0:
        print("[STDERR]", r.stderr.strip()[-1500:])
    else:
        print("[OK] 退出码 0")
    print()
    os.unlink(path)
