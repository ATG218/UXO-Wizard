"""
Pre-initialization hook for rpy2 to prevent R_HOME conflicts in PyInstaller
"""
import os
import sys

def patch_rpy2_initialization():
    """Patch rpy2 initialization to use packaged R environment"""
    
    # Only apply patch if we're in PyInstaller bundle
    if not hasattr(sys, '_MEIPASS'):
        return
    
    try:
        import rpy2.rinterface as rinterface
        
        # Store original _setrenvvars function
        original_setrenvvars = getattr(rinterface, '_setrenvvars', None)
        
        if original_setrenvvars:
            def patched_setrenvvars(action):
                """Patched version that skips problematic environment variable updates"""
                try:
                    # Only allow specific safe environment variables
                    safe_vars = {
                        'R_HOME', 'R_LIBS_USER', 'R_ENVIRON_USER', 
                        'R_PROFILE_USER', 'R_DOC_DIR', 'R_INCLUDE_DIR', 'R_SHARE_DIR'
                    }
                    
                    # Get current environment before calling original
                    current_env = dict(os.environ)
                    
                    # Call original function but catch any errors
                    try:
                        original_setrenvvars(action)
                    except (OSError, ValueError) as e:
                        print(f"rpy2 environment setup warning: {e}")
                        # Restore our safe environment
                        for key in safe_vars:
                            if key in current_env:
                                os.environ[key] = current_env[key]
                        
                except Exception as e:
                    print(f"Failed to patch rpy2 initialization: {e}")
            
            # Apply the patch
            rinterface._setrenvvars = patched_setrenvvars
            print("Applied rpy2 initialization patch")
            
    except ImportError:
        # rpy2 not available yet, patch will be applied when imported
        pass

# Apply patch immediately
patch_rpy2_initialization()