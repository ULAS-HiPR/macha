import pytest


def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from config import MachaConfig, load_config
        from task import Task, TaskManager
        from database import DatabaseManager
        from logger import setup_logger
        from metrics_task import MetricsTask
        from camera_task import CameraTask
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_sensor_task_imports():
    """Test that sensor task modules can be imported."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        
        from baro_task import BaroTask
        from imu_task import ImuTask
        assert True
    except ImportError as e:
        pytest.fail(f"Sensor task import failed: {e}")


def test_config_validation():
    """Test basic config validation."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from config import BarometerParameters, ImuParameters
    
    # Test valid barometer parameters
    baro_params = BarometerParameters(
        i2c_bus=1,
        address=0x77,
        sea_level_pressure=1013.25
    )
    assert baro_params.i2c_bus == 1
    assert baro_params.address == 0x77
    
    # Test valid IMU parameters
    imu_params = ImuParameters(
        i2c_bus=1,
        address=0x6A,
        accel_range="4G",
        gyro_range="500DPS"
    )
    assert imu_params.i2c_bus == 1
    assert imu_params.address == 0x6A
    assert imu_params.accel_range == "4G"
    assert imu_params.gyro_range == "500DPS"


def test_invalid_config():
    """Test that invalid configurations raise errors."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from config import BarometerParameters, ImuParameters
    from pydantic import ValidationError
    
    # Test invalid barometer address
    with pytest.raises(ValidationError):
        BarometerParameters(address=0x50)  # Invalid address
    
    # Test invalid IMU ranges
    with pytest.raises(ValidationError):
        ImuParameters(accel_range="32G")  # Invalid range
    
    with pytest.raises(ValidationError):
        ImuParameters(gyro_range="3000DPS")  # Invalid range