import io, subprocess

path = r'第5章 朗之万动力学与得分函数/5.2 得分函数：对数概率的梯度.md'

# 从 git HEAD 取原始内容，定位那条归一化常数公式的正确单行
head = subprocess.check_output(['git','show',f'HEAD:{path}']).decode('utf-8')
target = None
for l in head.split('\n'):
    if l.startswith('$$s(x) =') and 'log p(x)' in l and '\\frac{\\tilde{p}(x)}{Z}' in l:
        target = l
        break
print('HEAD 正确单行:', repr(target))

# 当前文件
cur = io.open(path, encoding='utf-8').read()
cur_lines = cur.split('\n')

# 找到被拆坏的多行序列：以 '$$s(x) = ' 开头、后续行以 'abla' 起始、以 '$$' 结尾
start = None
for i,l in enumerate(cur_lines):
    if l == '$$s(x) = ' and i+1 < len(cur_lines) and cur_lines[i+1].startswith('abla'):
        start = i
        break
print('损坏起始行索引(0基):', start)

if start is not None:
    end = start
    while end < len(cur_lines) and not cur_lines[end].rstrip().endswith('$$'):
        end += 1
    print('损坏结束行索引:', end, ' 内容片段:', cur_lines[end][:30])
    broken = '\n'.join(cur_lines[start:end+1])
    print('损坏块:', repr(broken))
    fixed = cur.replace(broken, target)
    io.open(path,'w',encoding='utf-8').write(fixed)
    print('已修复')

# 复核：确认无残留 'abla' 行首、且 \\nabla 计数增加
chk = io.open(path, encoding='utf-8').read()
chk_lines = chk.split('\n')
print('残留 abla 行首:', sum(1 for l in chk_lines if l.startswith('abla')))
print('nabla 总数:', chk.count('\\nabla'))
print('总行数:', len(chk_lines))
