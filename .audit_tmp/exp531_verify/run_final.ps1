$env:PYTHONIOENCODING = "utf-8"
$py = "D:\works\deepinv\.audit_tmp\venv042\Scripts\python.exe"
$script = "d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.3-1\5.3-1.py"
$out = "d:\works\deepinv\.audit_tmp\exp531_verify\run_final.log"
$err = "d:\works\deepinv\.audit_tmp\exp531_verify\run_final.err"
$p = Start-Process -FilePath $py -ArgumentList "`"$script`"" -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait -PassThru
"EXITCODE=$($p.ExitCode)"
