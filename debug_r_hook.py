"""
Debug hook to check R environment and rpy2 status
"""
import os
import sys

def debug_r_environment():
    """Debug R environment setup in packaged app"""
    print("=== R Environment Debug ===")
    
    # Check if we're in a PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        print(f"Running in PyInstaller bundle: {sys._MEIPASS}")
    else:
        print("Not running in PyInstaller bundle")
    
    # Set R_HOME if not set
    r_home = os.environ.get('R_HOME')
    print(f"R_HOME before: {r_home}")
    
    if not r_home:
        # Try to get R_HOME from R itself
        import subprocess
        try:
            result = subprocess.run(['R', 'RHOME'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                r_home = result.stdout.strip()
                os.environ['R_HOME'] = r_home
                print(f"Set R_HOME to: {r_home}")
            else:
                print(f"Could not get R_HOME: {result.stderr}")
        except Exception as e:
            print(f"Failed to get R_HOME: {e}")
    
    print(f"R_HOME after: {os.environ.get('R_HOME')}")
    
    # Try to find R executable
    import subprocess
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("R executable found and working")
            print(f"R version: {result.stdout.split()[2] if len(result.stdout.split()) > 2 else 'unknown'}")
        else:
            print(f"R executable failed: {result.stderr}")
    except Exception as e:
        print(f"Could not run R executable: {e}")
    
    # Try to import rpy2 with error handling
    try:
        # Pre-configure environment to prevent initialization errors
        if hasattr(sys, '_MEIPASS'):
            # Ensure R_HOME is locked before rpy2 import
            r_home = os.environ.get('R_HOME')
            if r_home:
                os.environ['RPY2_R_HOME'] = r_home
        
        import rpy2
        try:
            version = rpy2.__version__
            print(f"rpy2 imported successfully: {version}")
        except AttributeError:
            # Try alternative way to get version
            try:
                from rpy2._version import version
                print(f"rpy2 imported successfully: {version}")
            except:
                print("rpy2 imported successfully (version unknown)")
        
        try:
            import rpy2.robjects as ro
            print("rpy2.robjects imported successfully")
            
            try:
                r = ro.r
                print("R interface created successfully")
                
                # Try a simple R command
                result = r('1 + 1')
                print(f"Simple R command (1+1) result: {result}")
                
                # Test specific GPR imports
                print("Testing GPR-specific imports...")
                try:
                    from src.processing.scripts.gpr.utils.LoadGPRdata import ProjectGPRdata
                    print("✅ LoadGPRdata import successful")
                except Exception as e:
                    print(f"❌ LoadGPRdata import failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Test RGPR package loading in R
                print("Testing RGPR package loading in R...")
                try:
                    rgpr_result = r('library(RGPR)')
                    print("✅ RGPR library loaded successfully in R")
                    
                    # Test a basic RGPR function
                    rgpr_version = r('packageVersion("RGPR")')
                    print(f"✅ RGPR version: {rgpr_version}")
                    
                    # Test if basic RGPR functions are available
                    rgpr_functions = r('ls("package:RGPR")')
                    print(f"✅ RGPR has {len(rgpr_functions)} functions available")
                    
                except Exception as e:
                    print(f"❌ RGPR library loading failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    from src.processing.scripts.gpr.utils.GPRdataSegmentProcessor import GPRdataSegmentProcessor
                    print("✅ GPRdataSegmentProcessor import successful")
                except Exception as e:
                    print(f"❌ GPRdataSegmentProcessor import failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    from src.processing.scripts.gpr.utils.gpr_configs import ProcessingConfig
                    print("✅ ProcessingConfig import successful")
                except Exception as e:
                    print(f"❌ ProcessingConfig import failed: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"Failed to create R interface: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Failed to import rpy2.robjects: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Failed to import rpy2: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== End R Environment Debug ===")

debug_r_environment()