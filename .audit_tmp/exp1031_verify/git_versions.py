# -*- coding: utf-8 -*-
"""提取实验10.3-1各git版本的脚本字节，检查语法与关键段落差异"""
import subprocess, ast, sys, os

REPO = r"d:\works\deepinv"
P = "第10章 层级VAE与扩散的变分推导/第十章 配套实验/实验10.3-1/10.3-1.py"
P2 = "第10章 层级VAE与扩散的变分推导/第十章配套实验/实验10.3-1/10.3-1.py"
OUT = r"d:\works\deepinv\.audit_tmp\exp1031_verify"

import hashlib
for ref in ["accb9e7", "29e1368", "541068b", "a87d11e", "HEAD"]:
    for p in [P, P2]:
        r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{p}"], capture_output=True)
        if r.returncode != 0:
            print(ref, "NOT IN REF", p.split('/')[-1])
            continue
        txt = r.stdout.decode("utf-8", "replace")
        name = p.split('/')[-1]
        fp = os.path.join(OUT, f"{ref}_{name}")
        with open(fp, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        try:
            ast.parse(txt)
            print(ref, name, "SYNTAX OK lines=", len(txt.splitlines()))
        except SyntaxError as e:
            print(ref, name, "SyntaxError line", e.lineno, ":", (e.text or "").strip())
