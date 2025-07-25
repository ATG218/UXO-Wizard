import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import os
import numpy as np

print("--- R Environment Diagnostic ---")

# 1. Check R_HOME environment variable
r_home = os.environ.get('R_HOME')
print(f"\n[1] R_HOME environment variable: {r_home if r_home else 'Not Set'}")
if not r_home:
    print("    -> Recommendation: Set the R_HOME environment variable to your R installation directory.")

# 2. Check basic R interface
try:
    r_version_str = robjects.r('R.version.string')[0]
    print(f"\n[2] Successfully connected to R. Version: {r_version_str}")
except Exception as e:
    print(f"\n[2] FAILED to connect to R. rpy2 may not be able to find your R installation.")
    print(f"    Error: {e}")
    print("--- End of Diagnostic ---")
    exit()

# 3. Check R library paths
try:
    lib_paths = robjects.r('.libPaths()')
    print("\n[3] R is looking for packages in these directories (.libPaths):")
    for path in lib_paths:
        print(f"    - {path}")
except Exception as e:
    print(f"\n[3] FAILED to get R library paths. Error: {e}")

# 4. Check if RGPR is installed
try:
    utils = importr('utils')
    installed_packages = utils.installed_packages()
    # Convert to a numpy array to access columns. The first column contains package names.
    np_packages = np.array(installed_packages)
    pkg_names = np_packages[:]

    if 'RGPR' in pkg_names:
        print("\n[4] Found 'RGPR' in the list of installed R packages.")
    else:
        print("\n[4] 'RGPR' package NOT FOUND in the list of installed R packages.")
        print("    -> Recommendation: Install RGPR in your R environment using: install.packages('RGPR')")
        print("--- End of Diagnostic ---")
        exit()

except Exception as e:
    print(f"\n[4] FAILED to check installed packages. Error: {e}")
    print("--- End of Diagnostic ---")
    exit()

# 5. Try to load RGPR
try:
    print("\n[5] Attempting to load 'RGPR' package...")
    # Using robjects.r('library(...)') is often more robust for diagnostics
    robjects.r('library(RGPR)')
    print("    -> Successfully executed library(RGPR) command.")
    print("       If there were no errors above, the package is loaded correctly in R.")

except Exception as e:
    print(f"\n[5] FAILED to load 'RGPR' package.")
    print(f"    This is likely the source of the error in the application.")
    print(f"    Error details: {e}")
    print("\n    -> Recommendation: Try running `library(RGPR)` directly in an R console.")
    print("       Look for any error messages or warnings that appear there.")

print("\n--- End of Diagnostic ---") 