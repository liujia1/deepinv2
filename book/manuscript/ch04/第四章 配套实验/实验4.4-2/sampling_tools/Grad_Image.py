
#    Compute the gradient operator of an image
#
#    Copyright (C) 2023 MI2G
#    Dobson, Paul pdobson@ed.ac.uk
#    Kemajou, Mbakam Charlesquin cmk2000@hw.ac.uk
#    Klatzer, Teresa t.klatzer@sms.ed.ac.uk
#    Melidonis, Savvas sm2041@hw.ac.uk
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch

# horizontal and vertical finite defference operators
def diffh(x):
    # zeros_like 保证与 x 同设备/同 dtype（原 torch.zeros 在 GPU 上会生成 CPU 张量）
    x_diffh = torch.zeros_like(x)
    x_diffh[:,:-1] = x[:,1:] - x[:,0:-1]
    return x_diffh

def diffv(x):
    x_diffv = torch.zeros_like(x)
    x_diffv[:-1, :] = x[1:, :] - x[0:-1,:]
    # 修复（2026-10复核）：原实现末尾多了 .T 转置，使各向同性TV在 sqrt 内把
    # 水平差 h[j,i] 与竖直差 v[i,j] 错配（正确应为 v[j,i]）。该转置不改变
    # 各分量平方和，但改变 sqrt 内配对，TV 偏差实测可达 2.5%~14%，并经
    # 均衡关系 theta*=d/E[TV] 同比例污染 SAPG 的 theta 估计。
    return x_diffv

# Total variation norm（零填充前向差分，与 chambolle_prox_TV.GradientIm 同约定）
def TVnorm(x):
    y = torch.sum(torch.sqrt(diffh(x)**2 + diffv(x)**2))
    return y

def Grad_Image(x, device):

    with torch.no_grad():

        x = x.to(device).clone()
        x_temp = x[1:, :] - x[0:-1,:]
        dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
        dux = dux.T
        x_temp = x[:,1:] - x[:,0:-1]
        duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
        return  torch.cat((dux,duy),dim=0).to(device)
