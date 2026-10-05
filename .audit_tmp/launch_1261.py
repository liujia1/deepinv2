# -*- coding: utf-8 -*-
import runpy, sys, os
p = r'd:\works\deepinv\第12章 Score ≡ ELBO：殊途同归\第十二章 配套实验\实验12.6-1\12.6-1.py'
os.chdir(os.path.dirname(p))
sys.argv = [p]
runpy.run_path(p, run_name='__main__')
