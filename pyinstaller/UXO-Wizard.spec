# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

try:
    pyqtlet2_datas, pyqtlet2_binaries, pyqtlet2_hiddenimports = collect_all("pyqtlet2")
except Exception:
    pyqtlet2_datas, pyqtlet2_binaries, pyqtlet2_hiddenimports = [], [], []

try:
    rpy2_datas, rpy2_binaries, rpy2_hiddenimports = collect_all("rpy2")
except Exception:
    rpy2_datas, rpy2_binaries, rpy2_hiddenimports = [], [], []

env_dir = Path.cwd() / "build" / "ux0-env"
env_binaries = [
    (str(env_dir / "bin" / "R"),                     "ux0-env/bin"),
    (str(env_dir / "bin" / "Rscript"),               "ux0-env/bin"),
    (str(env_dir / "lib" / "R" / "lib" / "libR.dylib"), "ux0-env/lib/R/lib"),
    # Add the actual OpenBLAS library and related BLAS/LAPACK libraries
    (str(env_dir / "lib" / "libopenblas.0.dylib"),   "ux0-env/lib/libopenblas.0.dylib"),
    (str(env_dir / "lib" / "libopenblasp-r0.3.30.dylib"), "ux0-env/lib/libopenblasp-r0.3.30.dylib"),
    # Create the missing libRblas.dylib that rpy2 is looking for at root level
    (str(env_dir / "lib" / "libopenblas.0.dylib"),   "libRblas.dylib"),
    (str(env_dir / "lib" / "libopenblas.0.dylib"),   "libRlapack.dylib"),
    # Also place them in the R/lib directory
    (str(env_dir / "lib" / "libopenblas.0.dylib"),   "ux0-env/lib/R/lib/libRblas.dylib"),
    (str(env_dir / "lib" / "libopenblas.0.dylib"),   "ux0-env/lib/R/lib/libRlapack.dylib"),
]

env_datas = [
    # Only include essential R components that actually exist
    (str(env_dir / "lib" / "R"), "ux0-env/lib/R"),
]

# ───────────────────────────── Analysis section ──────────────────────────
a = Analysis(
    ["../main.py"],
    pathex=[],
    binaries=[
        *pyqtlet2_binaries,
        *rpy2_binaries,
        *env_binaries,          
    ],
    datas=[
        ("../src", "src"),
        ("../src/ui/map/layer_types.py",      "src/ui/map"),
        ("../src/ui/assets/icon.png",         "src/ui/assets"),
        *pyqtlet2_datas,
        *rpy2_datas,
        *env_datas,             
    ],
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtPrintSupport",
        "PySide6.QtWebChannel",
        "jaraco.text",
        "jaraco.functools",
        "jaraco.context",
        "autocommand",
        "more-itertools",
        "pyqtlet2",
        "qtpy",
        "qtpy.QtCore",
        "qtpy.QtGui",
        "qtpy.QtWidgets",
        "qtpy.QtWebEngineWidgets",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_agg",
        "plotly",
        "seaborn",
        "scikit-learn",
        "scikit-image",
        "geopandas",
        "shapely",
        "fiona",
        "rasterio",
        "pyproj",
        "openpyxl",
        "xlrd",
        "h5py",
        "netcdf4",
        "requests",
        "lxml",
        "tifffile",
        "imageio",
        "imagecodecs",
        "utm",
        "contextily",
        "geopy",
        "geographiclib",
        "mercantile",
        "xyzservices",
        "pyarrow",
        "fastparquet",
        "fsspec",
        "cramjam",
        "obspy",
        "pywavelets",
        "joblib",
        "threadpoolctl",
        "networkx",
        "rpy2",
        "rpy2.rinterface",
        "rpy2.rinterface_lib",
        "rpy2.rinterface_lib.embedded",
        "rpy2.rinterface_lib.callbacks",
        "rpy2.rinterface_lib.conversion",
        "rpy2.robjects",
        "rpy2.robjects.numpy2ri",
        "rpy2.robjects.conversion",
        "rpy2.robjects.packages",
        "rpy2._version",
        "zstandard",
        "brotli",
        "lz4",
        "snappy",
        "loguru",
        "psutil",
        "click",
        "colorama",
        "tqdm",
        "jinja2",
        "markupsafe",
        "pygments",
        "datetime",
        "pathlib",
        "json",
        "csv",
        *pyqtlet2_hiddenimports,
        *rpy2_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        "pyinstaller/set_conda_env.py",   
        "pyinstaller/rpy2_init_hook.py",  
        "debug_r_hook.py",   
        "pyinstaller/set_scripts_dir.py",             
    ],
    excludes=[
        "pytest",
        "sphinx",
        "wheel",
        "pip",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="UXO-Wizard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,               
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["../src/ui/assets/icon.png"],
)

app = BUNDLE(
    exe,
    name="UXO-Wizard.app",
    icon="../src/ui/assets/icon.png",
    bundle_identifier=None,
)