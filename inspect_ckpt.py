import torch

base = r'd:\works\deepinv\第17章 自监督学习与等变架构\第十七章 参考实验\实验17.3 等变成像与测量一致性'
for name in ['ckpt_Naive.pt', 'ckpt_EI.pt', 'ckpt_Supervised.pt']:
    path = base + '\\' + name
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(name, '->', list(ckpt.keys()))
    print('  epoch:', ckpt.get('epoch'))
    print('  is_final:', ckpt.get('is_final'))
    state = ckpt.get('model_state', ckpt.get('model_state_dict'))
    print('  state num_keys:', len(state))
    print('  first 3 keys:', list(state.keys())[:3])
    print('  first 3 shapes:', [state[k].shape for k in list(state.keys())[:3]])
    print()
