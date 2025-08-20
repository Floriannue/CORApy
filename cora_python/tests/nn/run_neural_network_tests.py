#!/usr/bin/env python3
"""
Test runner for Neural Network tests

This script runs all the neural network tests and provides a summary of results.
"""

import sys
import os
import pytest
import subprocess
from pathlib import Path

def run_tests_with_pytest():
    """Run tests using pytest directly"""
    print("Running Neural Network tests with pytest...")
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent / "neuralNetwork"
    
    # Run pytest on the neuralNetwork test directory
    result = pytest.main([
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes"  # Colored output
    ])
    
    return result

def run_tests_with_subprocess():
    """Run tests using subprocess (alternative method)"""
    print("Running Neural Network tests with subprocess...")
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent / "neuralNetwork"
    
    # Run pytest using subprocess
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True, check=False)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main function"""
    print("=" * 60)
    print("Neural Network Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not (Path(__file__).parent / "neuralNetwork").exists():
        print("Error: neuralNetwork test directory not found!")
        print("Please run this script from the tests/nn directory.")
        return 1
    
    # Try running tests with pytest first
    try:
        result = run_tests_with_pytest()
        return result
    except Exception as e:
        print(f"Error with pytest: {e}")
        print("Falling back to subprocess method...")
        
        try:
            result = run_tests_with_subprocess()
            return result
        except Exception as e2:
            print(f"Error with subprocess: {e2}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
