import sys

packages = ['sen2sr', 'mlstac', 'opensr', 'opensr_test', 'torch', 'rasterio', 'scipy', 'sklearn']
for pkg in packages:
    try:
        m = __import__(pkg)
        ver = getattr(m, '__version__', 'installed (no version attr)')
        path = getattr(m, '__file__', 'unknown')
        print(f"{pkg}: {ver} | {path}")
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})")
