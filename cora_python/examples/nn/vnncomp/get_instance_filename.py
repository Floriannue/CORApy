"""
get_instance_filename - create a filename for storing the parsed neural 
    network and vnnlib specification

Syntax:
    instance_filename, model_name, vnnlib_name = get_instance_filename(
        bench_name, model_path, vnnlib_path)

Inputs:
    bench_name - name of the benchmark
    model_path - path to the .onnx-file
    vnnlib_path - path to the .vnnlib-file

Outputs:
    instance_filename - filename (unique for this instance)
    model_name - name of the .onnx-file
    vnnlib_name - name of the .vnnlib-file

References:
    [1] VNN-COMP'24

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import re
from typing import Tuple


def get_instance_filename(bench_name: str, model_path: str, vnnlib_path: str) -> Tuple[str, str, str]:
    """
    Create a filename for storing parsed network and specification.
    
    Args:
        bench_name: Name of the benchmark
        model_path: Path to .onnx file
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Tuple of (instance_filename, model_name, vnnlib_name)
    """
    # Extract model name from path (everything before .onnx, possibly followed by .gz)
    model_match = re.search(r'([^/\\]+)(?=\.onnx(?:\.gz)?$)', model_path)
    model_name = model_match.group(1) if model_match else 'unknown_model'
    
    # Extract vnnlib name from path (everything before .vnnlib, possibly followed by .gz)
    vnnlib_match = re.search(r'([^/\\]+)(?=\.vnnlib(?:\.gz)?$)', vnnlib_path)
    vnnlib_name = vnnlib_match.group(1) if vnnlib_match else 'unknown_vnnlib'
    
    # Create unique instance filename
    instance_filename = f'{bench_name}_{model_name}_{vnnlib_name}.pkl'
    
    return instance_filename, model_name, vnnlib_name

