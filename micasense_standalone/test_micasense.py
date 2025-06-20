#!/usr/bin/env python3
"""
Test script to verify MicaSense module installation
"""

def test_micasense_import():
    """Test that all micasense modules can be imported"""
    try:
        print("Testing MicaSense module import...")
        
        # Test main module
        import micasense
        print(f"✅ micasense module imported successfully (version: {micasense.__version__})")
        
        # Test individual modules
        from micasense import capture
        print("✅ capture module imported")
        
        from micasense import image
        print("✅ image module imported")
        
        from micasense import imageset
        print("✅ imageset module imported")
        
        from micasense import utils
        print("✅ utils module imported")
        
        from micasense import plotutils
        print("✅ plotutils module imported")
        
        from micasense import imageutils
        print("✅ imageutils module imported")
        
        from micasense import metadata
        print("✅ metadata module imported")
        
        from micasense import panel
        print("✅ panel module imported")
        
        from micasense import dls
        print("✅ dls module imported")
        
        print("\n🎉 All MicaSense modules imported successfully!")
        print("Your environment is ready to use MicaSense image processing!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def show_python_info():
    """Show current Python environment information"""
    import sys
    import os
    
    print("\n" + "="*50)
    print("PYTHON ENVIRONMENT INFO")
    print("="*50)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0]}")
    
    # Check if we're in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"Conda environment: {conda_env}")
    
    # Show installed packages related to image processing
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except:
        print("NumPy: Not installed")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except:
        print("OpenCV: Not installed")
    
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except:
        print("Matplotlib: Not installed")

if __name__ == "__main__":
    show_python_info()
    test_micasense_import() 