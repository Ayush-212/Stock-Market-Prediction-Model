import importlib
modules = ['numpy', 'pandas', 'yfinance',
           'matplotlib', 'sklearn', 'tensorflow', 'arch']
for m in modules:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', getattr(mod, 'VERSION', 'unknown'))
        print(f"{m}: OK ({v})")
    except Exception as e:
        print(f"{m}: ERROR ({e})")
