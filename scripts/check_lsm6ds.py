#!/usr/bin/env python3
"""
Script to check available LSM6DS classes in the adafruit_lsm6ds library
"""

import sys

def check_lsm6ds_library():
    """Check what LSM6DS classes are available."""
    print("=== LSM6DS Library Inspector ===")
    
    try:
        import adafruit_lsm6ds
        print("✓ adafruit_lsm6ds library imported successfully")
        print(f"Library location: {adafruit_lsm6ds.__file__}")
        print()
        
        # List all attributes in the module
        print("Available attributes in adafruit_lsm6ds:")
        attrs = dir(adafruit_lsm6ds)
        
        classes = []
        enums = []
        other = []
        
        for attr in attrs:
            if attr.startswith('_'):
                continue
            
            obj = getattr(adafruit_lsm6ds, attr)
            if isinstance(obj, type):
                classes.append(attr)
            elif hasattr(obj, '__members__'):  # Enum-like
                enums.append(attr)
            else:
                other.append(attr)
        
        if classes:
            print("\nSensor Classes:")
            for cls in sorted(classes):
                print(f"  - {cls}")
        
        if enums:
            print("\nEnumerations:")
            for enum in sorted(enums):
                print(f"  - {enum}")
                try:
                    enum_obj = getattr(adafruit_lsm6ds, enum)
                    if hasattr(enum_obj, '__members__'):
                        members = list(enum_obj.__members__.keys())
                        print(f"    Members: {', '.join(members[:5])}")
                        if len(members) > 5:
                            print(f"    ... and {len(members) - 5} more")
                except:
                    pass
        
        if other:
            print("\nOther attributes:")
            for attr in sorted(other):
                print(f"  - {attr}")
        
        # Test LSM6DS class variants
        print("\n=== Testing LSM6DS Class Availability ===")
        test_classes = [
            'LSM6DSOX',
            'LSM6DS33', 
            'LSM6DS3TRC',
            'LSM6DSO32',
            'LSM6DS',
            'LSM6DSOX',
            'LSM6DSL',
            'LSM6DS3',
        ]
        
        available_classes = []
        for cls_name in test_classes:
            if hasattr(adafruit_lsm6ds, cls_name):
                cls_obj = getattr(adafruit_lsm6ds, cls_name)
                if isinstance(cls_obj, type):
                    available_classes.append(cls_name)
                    print(f"✓ {cls_name} - Available")
                else:
                    print(f"? {cls_name} - Found but not a class")
            else:
                print(f"✗ {cls_name} - Not found")
        
        if available_classes:
            print(f"\nRecommended sensor class: {available_classes[0]}")
        else:
            print("\n⚠️  No sensor classes found!")
        
        return available_classes
        
    except ImportError as e:
        print(f"✗ Failed to import adafruit_lsm6ds: {e}")
        print("\nTo install:")
        print("  pip install adafruit-circuitpython-lsm6ds")
        print("  or")
        print("  uv add adafruit-circuitpython-lsm6ds")
        return []
    except Exception as e:
        print(f"✗ Error inspecting library: {e}")
        return []

def test_i2c_connection():
    """Test I2C connection to LSM6DS sensor."""
    print("\n=== Testing I2C Connection ===")
    
    try:
        import board
        import busio
        print("✓ Board and busio imported")
        
        i2c = busio.I2C(board.SCL, board.SDA)
        print("✓ I2C bus initialized")
        
        # Scan for devices
        while not i2c.try_lock():
            pass
        
        try:
            devices = i2c.scan()
            print(f"I2C devices found: {[hex(d) for d in devices]}")
            
            # Check for common LSM6DS addresses
            lsm6ds_addresses = [0x6A, 0x6B]
            found_lsm6ds = []
            
            for addr in lsm6ds_addresses:
                if addr in devices:
                    found_lsm6ds.append(hex(addr))
            
            if found_lsm6ds:
                print(f"✓ Potential LSM6DS devices at: {', '.join(found_lsm6ds)}")
            else:
                print("✗ No LSM6DS devices found at expected addresses (0x6A, 0x6B)")
                
        finally:
            i2c.unlock()
            i2c.deinit()
            
    except Exception as e:
        print(f"✗ I2C test failed: {e}")

def main():
    """Main function."""
    available_classes = check_lsm6ds_library()
    test_i2c_connection()
    
    print("\n=== Summary ===")
    if available_classes:
        print(f"Available sensor classes: {', '.join(available_classes)}")
        print(f"Use this in your code: adafruit_lsm6ds.{available_classes[0]}")
    else:
        print("No sensor classes available. Check library installation.")
    
    return 0 if available_classes else 1

if __name__ == "__main__":
    sys.exit(main())