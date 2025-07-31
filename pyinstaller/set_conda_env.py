import os, sys, pathlib, site

root = pathlib.Path(sys._MEIPASS) / "ux0-env"   # temp dir PyInstaller creates
os.environ["CONDA_PREFIX"] = str(root)
os.environ["PATH"]         = f"{root}/bin:" + os.environ["PATH"]
os.environ["R_HOME"]       = f"{root}/lib/R"
os.environ["R_LIBS_USER"]  = f"{root}/lib/R/library"

# Prevent rpy2 from overriding our R_HOME
os.environ["RPY2_R_HOME_OVERRIDE"] = "0"
os.environ["R_ENVIRON_USER"] = f"{root}/lib/R/etc/Renviron"

# Set additional R environment variables to prevent conflicts
os.environ["R_PROFILE_USER"] = f"{root}/lib/R/etc/Rprofile.site"
os.environ["R_DOC_DIR"] = f"{root}/lib/R/doc"
os.environ["R_INCLUDE_DIR"] = f"{root}/lib/R/include"
os.environ["R_SHARE_DIR"] = f"{root}/lib/R/share"

site.addsitedir(f"{root}/lib/python3.13/site-packages")