import os
import pathlib
import subprocess
import sys

TARGET = pathlib.Path(
    r"h:\deepinv\deepinv2\第15章 扩散模型的架构实践：UNet → DiT"
    r"\第十五章 配套实验\实验15.4-1\15.4-1.py"
)
LOG = pathlib.Path(r"h:\deepinv\deepinv2\.audit_tmp\exp1541_rerun.log")

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

with LOG.open("w", encoding="utf-8", errors="replace") as f:
    p = subprocess.run(
        [sys.executable, str(TARGET)],
        cwd=str(TARGET.parent),
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
    )
print("returncode:", p.returncode)
