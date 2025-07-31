# Sets UXO_SCRIPTS_DIR env var to "<executables directory>/scripts"
import os, sys, pathlib
exe_dir = pathlib.Path(sys.executable).resolve().parent
scripts_dir = exe_dir / "scripts"
os.environ["UXO_SCRIPTS_DIR"] = str(scripts_dir)