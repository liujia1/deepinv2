import json, os, torch
EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"
def hist(p):
    ck = torch.load(os.path.join(EXP,p), map_location="cpu", weights_only=False)
    return [float(v) for v in ck["history"]["loss"]]
he = hist("ddpm_epsilon_checkpoint.pth"); hx = hist("ddpm_x0_checkpoint.pth")
res = {
 "epsilon_prediction": {"final_loss": he[-1], "loss_curve": he},
 "x0_prediction": {"final_loss": hx[-1], "loss_curve": hx},
 "T": 200, "num_epochs": 50,
}
with open(os.path.join(EXP,"results_summary.json"),"w",encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("written; eps_final=%.6f x0_final=%.6f len=%d/%d" % (he[-1], hx[-1], len(he), len(hx)))
