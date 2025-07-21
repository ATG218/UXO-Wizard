#!/usr/bin/env python3
"""
Test script to verify that the layer system import fix works
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_layer_system_imports():
    """Test if the layer system imports work with the new direct import method"""
    print("Testing layer system imports...")
    
    try:
        # Test the direct import method used in base.py
        import importlib.util
        
        # Get the absolute path to layer_types.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        layer_types_path = os.path.join(current_dir, 'src', 'ui', 'map', 'layer_types.py')
        layer_types_path = os.path.normpath(layer_types_path)
        
        print(f"Attempting to load layer_types from: {layer_types_path}")
        
        if not os.path.exists(layer_types_path):
            print(f"ERROR: layer_types.py not found at {layer_types_path}")
            return False
        
        # Import layer_types module directly
        spec = importlib.util.spec_from_file_location("layer_types", layer_types_path)
        layer_types_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layer_types_module)
        
        # Extract the classes we need
        UXOLayer = layer_types_module.UXOLayer
        LayerType = layer_types_module.LayerType
        GeometryType = layer_types_module.GeometryType
        LayerStyle = layer_types_module.LayerStyle
        LayerSource = layer_types_module.LayerSource
        NORWEGIAN_CRS = layer_types_module.NORWEGIAN_CRS
        
        print("✓ Successfully imported layer_types directly")
        print(f"✓ UXOLayer: {UXOLayer}")
        print(f"✓ LayerType: {LayerType}")
        print(f"✓ Available layer types: {list(LayerType)}")
        print(f"✓ GeometryType: {GeometryType}")  
        print(f"✓ Available geometry types: {list(GeometryType)}")
        print(f"✓ LayerStyle: {LayerStyle}")
        print(f"✓ LayerSource: {LayerSource}")
        print(f"✓ NORWEGIAN_CRS type: {type(NORWEGIAN_CRS)}")
        
        # Test creating a simple layer style
        style = LayerStyle()
        print(f"✓ Created LayerStyle: color={style.point_color}, size={style.point_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        print(f"✗ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_layer_system_imports()
    if success:
        print("\n🎉 SUCCESS: Layer system imports work correctly!")
        print("The fix should resolve the layer creation issue.")
    else:
        print("\n💥 FAILED: Layer system imports still not working")
        print("Further investigation needed.")
    
    sys.exit(0 if success else 1)