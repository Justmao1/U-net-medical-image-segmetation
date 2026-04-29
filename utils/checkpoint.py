import torch


def load_networks(model, path):
    """加载模型权重，兼容不同格式的 checkpoint

    Args:
        model: PyTorch 模型实例
        path: checkpoint 文件路径，支持 {'state_dict': ...} 格式或直接 state_dict
    """
    state_dict = torch.load(path, map_location='cpu')
    print(f'Loading the model from {path}')

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # 兼容 {'state_dict': ...} 和直接 state_dict 两种格式
    if 'state_dict' in state_dict:
        sd = state_dict['state_dict']
    else:
        sd = state_dict

    net_state = model.state_dict()
    is_loaded = {n: False for n in net_state.keys()}

    for name, param in sd.items():
        if name in net_state:
            try:
                net_state[name].copy_(param)
                is_loaded[name] = True
            except Exception:
                print(f'While copying the parameter named [{name}], '
                      f'whose dimensions in the model are {list(net_state[name].shape)} and '
                      f'whose dimensions in the checkpoint are {list(param.shape)}.')
                raise RuntimeError
        else:
            print(f'Saved parameter named [{name}] is skipped')

    all_loaded = True
    for name in is_loaded:
        if not is_loaded[name]:
            print(f'Parameter named [{name}] is randomly initialized')
            all_loaded = False

    if all_loaded:
        print(f'All parameters are initialized using [{path}]')
