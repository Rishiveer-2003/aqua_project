"""
Pre-Flight Check Script
======================
Run this script BEFORE executing create_dynamic_dataset.py
to verify all dependencies and files are in place.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def print_section(text):
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)

def check_python_version():
    print_section("Python Version Check")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version must be 3.8 or higher")
        return False

def check_dependencies():
    print_section("Dependency Check")
    required_packages = [
        "pandas",
        "numpy",
        "joblib",
        "sklearn",
        "lightgbm",
        "xgboost",
        "geopy",
        "openmeteo_requests",
        "requests_cache",
        "retry_requests"
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            if package == "sklearn":
                import sklearn
                module = sklearn
            else:
                module = __import__(package)
            
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package:25} version: {version}")
        except ImportError:
            print(f"✗ {package:25} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_model_files():
    print_section("Model Files Check")
    required_files = [
        "lgbm_model.pkl",
        "rf_model.pkl",
        "xgboost_model.pkl",
        "svr_model.pkl",
        "knn_model.pkl"
    ]
    
    all_ok = True
    for filename in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"✓ {filename:25} ({size:.2f} MB)")
        else:
            print(f"✗ {filename:25} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_config_files():
    print_section("Configuration Files Check")
    
    if os.path.exists("feature_columns.json"):
        import json
        with open("feature_columns.json", "r") as f:
            features = json.load(f)
        print(f"✓ feature_columns.json      ({len(features)} features)")
        return True
    else:
        print("✗ feature_columns.json      NOT FOUND")
        return False

def check_script_file():
    print_section("Script File Check")
    
    if os.path.exists("create_dynamic_dataset.py"):
        size = os.path.getsize("create_dynamic_dataset.py") / 1024  # KB
        print(f"✓ create_dynamic_dataset.py ({size:.2f} KB)")
        return True
    else:
        print("✗ create_dynamic_dataset.py NOT FOUND")
        return False

def check_network():
    print_section("Network Connectivity Check")
    
    try:
        import urllib.request
        urllib.request.urlopen("https://api.open-meteo.com", timeout=5)
        print("✓ Open-Meteo API is reachable")
        network_ok = True
    except Exception as e:
        print(f"✗ Cannot reach Open-Meteo API: {e}")
        network_ok = False
    
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="test_app")
        location = geolocator.geocode("Mumbai", timeout=10)
        if location:
            print(f"✓ Nominatim geocoding is working (Mumbai: {location.latitude:.4f}, {location.longitude:.4f})")
        else:
            print("✗ Nominatim geocoding failed")
            network_ok = False
    except Exception as e:
        print(f"✗ Geocoding test failed: {e}")
        network_ok = False
    
    return network_ok

def test_model_loading():
    print_section("Model Loading Test")
    
    try:
        import joblib
        models = {}
        model_files = {
            'LightGBM': 'lgbm_model.pkl',
            'RandomForest': 'rf_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
            'SVR': 'svr_model.pkl',
            'KNN': 'knn_model.pkl',
        }
        
        for name, filename in model_files.items():
            if os.path.exists(filename):
                models[name] = joblib.load(filename)
                print(f"✓ Successfully loaded {name}")
            else:
                print(f"✗ Cannot load {name} (file missing)")
        
        if len(models) == 5:
            print(f"\n✓ All 5 models loaded successfully")
            return True
        else:
            print(f"\n✗ Only {len(models)}/5 models loaded")
            return False
    
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def estimate_runtime():
    print_section("Runtime Estimate")
    
    num_cities = 32
    days_per_city = 365
    seconds_per_city = 12  # Average processing time
    
    total_seconds = num_cities * seconds_per_city
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    print(f"Cities to process: {num_cities}")
    print(f"Days per city: {days_per_city}")
    print(f"Expected rows: ~{num_cities * (days_per_city - 7):,}")
    print(f"Estimated runtime: ~{minutes} minutes {seconds} seconds")
    print("\nNote: Actual runtime may vary based on network speed")

if __name__ == "__main__":
    print_header("PRE-FLIGHT CHECK FOR DATASET GENERATION")
    print("This script verifies all requirements before running create_dynamic_dataset.py")
    
    # Run all checks
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Model Files": check_model_files(),
        "Config Files": check_config_files(),
        "Script File": check_script_file(),
        "Network": check_network(),
        "Model Loading": test_model_loading()
    }
    
    # Runtime estimate
    estimate_runtime()
    
    # Final summary
    print_header("PRE-FLIGHT CHECK SUMMARY")
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("=" * 80)
        print("\nYou are ready to run the dataset generation script:")
        print("\n    python create_dynamic_dataset.py\n")
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        print("\nPlease fix the issues above before running create_dynamic_dataset.py")
        print("\nCommon fixes:")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • Missing files: Ensure all .pkl and .json files are in the directory")
        print("  • Network issues: Check your internet connection")
        print()
        sys.exit(1)
