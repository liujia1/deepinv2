import importlib

_submodules = [
    'autocor_plots',
    'blur_operators',
    'chambolle_prox_TV',
    'cshift',
    'Grad_Image',
    'max_eigenval',
    'measures',
    'tv_norm',
    'plots',
    'welford',
    'load_model',
    'spectral_normalize_chen',
]

for _mod in _submodules:
    try:
        _module = importlib.import_module(f'.{_mod}', __name__)
        _attrs = getattr(_module, '__all__', [name for name in dir(_module) if not name.startswith('_')])
        for _attr in _attrs:
            globals()[_attr] = getattr(_module, _attr)
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import sampling_tools.{_mod}: {e}")
