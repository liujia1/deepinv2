from PIL import Image
im = Image.open(r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.3-1\实验1_3_1_病态性.png')
W, H = im.size
print('size', im.size)
# 左半图 y 轴刻度区域
im.crop((0, 0, int(W*0.28), H)).resize((int(W*0.28)*2, H*2)).save(r'd:\works\deepinv\.audit_tmp\exp131_verify\crop_ylabel.png')
# 注释文本区域（中部偏右）
im.crop((int(W*0.28), int(H*0.35), int(W*0.52), int(H*0.75))).resize((int(W*0.24)*3, int(H*0.4)*3)).save(r'd:\works\deepinv\.audit_tmp\exp131_verify\crop_annot.png')
# 右图标题
im.crop((int(W*0.5), 0, W, int(H*0.12))).resize((int(W*0.5)*2, int(H*0.12)*2)).save(r'd:\works\deepinv\.audit_tmp\exp131_verify\crop_rtitle.png')
