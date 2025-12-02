"""
run_instance - run verification for a single VNN-COMP instance

Syntax:
    result_str, result_dict = run_instance(bench_name, model_path, vnnlib_path,
                                          results_path, timeout, verbose)

Inputs:
    bench_name - name of the benchmark
    model_path - path to the .onnx-file
    vnnlib_path - path to the .vnnlib-file
    results_path - path to the results file
    timeout - verification timeout in seconds
    verbose - print progress information

Outputs:
    result_str - result string ('unsat', 'sat', or 'unknown')
    result_dict - dictionary with result details

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sys
import os
import time
import gzip
import shutil
import tempfile
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

# Add cora_python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
from cora_python.contSet.polytope import Polytope
from cora_python.nn.nnHelper.validateNNoptions import validateNNoptions

# Import benchmark configuration
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_config import get_benchmark_options


def decompress_if_needed(file_path: str, verbose: bool = False) -> str:
    """
    Decompress a .gz file if needed, returning the path to the decompressed file.
    If the file is not compressed, returns the original path.
    
    This function handles cases where:
    - The file path ends with .gz (already compressed)
    - The file path doesn't end with .gz but the actual file is compressed (.gz exists)
    - The file is not compressed at all
    
    Args:
        file_path: Path to the file (may need .gz appended)
        verbose: Print progress information
        
    Returns:
        Path to the decompressed file (or original if not compressed)
    """
    # First, check if the file exists as-is
    if os.path.exists(file_path):
        # File exists, check if it's compressed
        if file_path.endswith('.gz'):
            # Need to decompress
            if verbose:
                print(f'Decompressing {file_path}...')
            
            # Create a temporary file for the decompressed content
            temp_dir = tempfile.gettempdir()
            base_name = os.path.basename(file_path).replace('.gz', '')
            temp_path = os.path.join(temp_dir, f'cora_vnncomp_{os.getpid()}_{base_name}')
            
            try:
                with gzip.open(file_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                if verbose:
                    print(f' done (temp file: {temp_path})')
                
                return temp_path
            except Exception as e:
                if verbose:
                    print(f'Error decompressing {file_path}: {e}')
                raise
        else:
            # File exists and is not compressed
            return file_path
    else:
        # File doesn't exist, try with .gz extension
        gz_path = file_path + '.gz'
        if os.path.exists(gz_path):
            # Found compressed version, decompress it
            if verbose:
                print(f'Found compressed file {gz_path}, decompressing...')
            
            # Create a temporary file for the decompressed content
            temp_dir = tempfile.gettempdir()
            base_name = os.path.basename(file_path)
            temp_path = os.path.join(temp_dir, f'cora_vnncomp_{os.getpid()}_{base_name}')
            
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                if verbose:
                    print(f' done (temp file: {temp_path})')
                
                return temp_path
            except Exception as e:
                if verbose:
                    print(f'Error decompressing {gz_path}: {e}')
                raise
        else:
            # File doesn't exist with or without .gz
            raise FileNotFoundError(f"File not found: {file_path} (also tried {gz_path})")


def convert_safe_set_to_union_unsafe_sets(specs):
    """
    Convert any safe set specifications to a union of unsafe sets.
    
    Args:
        specs: Specification or list of specifications
        
    Returns:
        List of specifications as unsafe sets
    """
    from cora_python.specification.specification import Specification
    
    # Handle single spec
    if not isinstance(specs, list):
        specs = [specs]
    
    unsafe_union_specs = []
    
    for spec in specs:
        if spec.type == 'safeSet':
            # Convert safe set to unsafe sets (one for each constraint)
            if hasattr(spec.set, 'A'):
                A, b = spec.set.A, spec.set.b
                # Invert each constraint
                for j in range(A.shape[0]):
                    # Invert the j-th constraint: A[j] @ y <= b[j] becomes -A[j] @ y <= -b[j]
                    Y_j = Polytope(-A[j:j+1, :], -b[j:j+1])
                    unsafe_union_specs.append(Specification(Y_j, 'unsafeSet'))
            elif hasattr(spec.set, 'c'):
                # Halfspace case
                Y_j = Polytope(-spec.set.c.reshape(1, -1), np.array([[-spec.set.d]]))
                unsafe_union_specs.append(Specification(Y_j, 'unsafeSet'))
        else:
            # Already unsafe set
            unsafe_union_specs.append(spec)
    
    return unsafe_union_specs


def spec_to_linear_constraint(spec):
    """
    Extract linear constraint from a specification.
    
    Args:
        spec: Specification object
        
    Returns:
        Tuple of (A, b, is_safe_set)
    """
    if hasattr(spec.set, 'c') and hasattr(spec.set, 'd'):
        # Halfspace
        A = spec.set.c.reshape(1, -1)
        b = np.array([[spec.set.d]])
    else:
        # Polytope
        A = spec.set.A
        b = spec.set.b
    
    is_safe_set = (spec.type == 'safeSet')
    
    return A, b, is_safe_set


def compute_criticality_of_specs(spec, ys):
    """
    Check the specification and compute how close we are to finding
    an adversarial example (< 0 means the specification is violated).
    
    Args:
        spec: Specification object
        ys: Output values (nOut x nSamples)
        
    Returns:
        Criticality value (scalar)
    """
    A, b, is_safe_set = spec_to_linear_constraint(spec)
    
    # Compute the logit difference
    ld_ys = A @ ys
    
    # Compute the criticality value per constraint
    cv_per_constr = ld_ys - b
    
    # Obtain the worst critical values
    if is_safe_set:
        # safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
        # Thus, unsafe if any(-A*y < -b)
        cv = np.min(-cv_per_constr)
    else:
        # unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
        # Thus, unsafe if all(A*y <= b)
        cv = np.max(cv_per_constr)
    
    return cv


def run_instance(bench_name: str, model_path: str, vnnlib_path: str,
                results_path: str, timeout: float = 120.0, verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Run verification for a single VNN-COMP instance.
    
    Args:
        bench_name: Name of benchmark (e.g., 'acasxu_2023')
        model_path: Path to .onnx file
        vnnlib_path: Path to .vnnlib file
        results_path: Path to write results
        timeout: Timeout in seconds
        verbose: Print progress
        
    Returns:
        Tuple of (result_str, result_dict) where result_str is 'unsat'/'sat'/'unknown'
    """
    if verbose:
        print(f'run_instance({bench_name},{model_path},{vnnlib_path},{results_path},{timeout},{verbose})...')
    
    # Initialize result
    result = {'str': 'unknown', 'time': -1, 'numVerified': 0}
    
    # Track decompressed files for cleanup
    temp_files = []
    
    # Store original paths for display and filename generation
    original_model_path = model_path
    original_vnnlib_path = vnnlib_path
    
    try:
        # Measure verification time
        total_time_start = time.time()
        
        # Get benchmark-specific options BEFORE loading network (use original paths for filename)
        from get_instance_filename import get_instance_filename
        _, model_name, vnnlib_name = get_instance_filename(bench_name, original_model_path, original_vnnlib_path)
        
        # Decompress files if needed
        decompressed_model_path = decompress_if_needed(original_model_path, verbose)
        if decompressed_model_path != original_model_path:
            temp_files.append(decompressed_model_path)
            model_path = decompressed_model_path
        else:
            model_path = original_model_path
        
        decompressed_vnnlib_path = decompress_if_needed(original_vnnlib_path, verbose)
        if decompressed_vnnlib_path != original_vnnlib_path:
            temp_files.append(decompressed_vnnlib_path)
            vnnlib_path = decompressed_vnnlib_path
        else:
            vnnlib_path = original_vnnlib_path
        
        try:
            bench_config = get_benchmark_options(bench_name, model_name, vnnlib_path)
            options = bench_config['options']
            permute_dims = bench_config.get('permute_dims', False)
            input_data_formats = bench_config.get('input_data_formats', 'BC')
            output_data_formats = bench_config.get('output_data_formats', '')
            target_network = bench_config.get('target_network', 'dlnetwork')
            contains_composite_layers = bench_config.get('contains_composite_layers', False)
            if verbose:
                print(f'Using benchmark-specific config for {bench_name}')
        except ValueError as e:
            if verbose:
                print(f'Warning: {e}')
                print('Using default options')
            # Fall back to default options
            options = {
                'nn': {
                    'use_approx_error': True,
                    'poly_method': 'bounds',
                    'train': {
                        'use_gpu': False,
                        'backprop': False,
                        'mini_batch_size': 1024
                    }
                }
            }
            permute_dims = False
            input_data_formats = 'BC'
            output_data_formats = ''
            target_network = 'dlnetwork'
            contains_composite_layers = False
        
        options = validateNNoptions(options, True)
        
        if verbose:
            print('--- Loading network...')
        
        # Load neural network with benchmark-specific parameters
        # MATLAB: nn = neuralNetwork.readONNXNetwork(modelPath, verbose, inputDataFormats, outputDataFormats, targetNetwork, containsCompositeLayers)
        nn = NeuralNetwork.readONNXNetwork(
            model_path,
            verbose,
            input_data_formats if input_data_formats else 'BC',
            output_data_formats if output_data_formats else 'BC',
            target_network,
            contains_composite_layers
        )
        
        if verbose:
            print(' done')
            print('--- Loading specification...')
        
        # Load specification
        X0, specs = vnnlib2cora(vnnlib_path)
        
        if verbose:
            print(' done')
        
        # Batch all input sets
        x_list = []
        r_list = []
        for X0_i in X0:
            xi = (X0_i.sup + X0_i.inf) / 2
            ri = (X0_i.sup - X0_i.inf) / 2
            
            # Handle permute_dims if needed (MATLAB: permute(reshape(xi,inSize),[2 1 3]))
            if permute_dims:
                # Get input size from first layer
                if hasattr(nn, 'layers') and len(nn.layers) > 0:
                    in_size = nn.layers[0].inputSize
                    if len(in_size) == 3:
                        # Reshape to [H, W, C], then permute to [W, H, C], then flatten
                        xi_reshaped = xi.reshape(in_size)
                        xi_permuted = np.transpose(xi_reshaped, (1, 0, 2))
                        xi = xi_permuted.flatten()
                        ri_reshaped = ri.reshape(in_size)
                        ri_permuted = np.transpose(ri_reshaped, (1, 0, 2))
                        ri = ri_permuted.flatten()
            
            x_list.append(xi.flatten())
            r_list.append(ri.flatten())
        
        x = np.column_stack(x_list) if len(x_list) > 1 else x_list[0]
        r = np.column_stack(r_list) if len(r_list) > 1 else r_list[0]
        
        # Convert safe sets to union of unsafe sets
        specs_list = convert_safe_set_to_union_unsafe_sets(specs)
        
        if verbose:
            print(f'--- Running verification on {len(specs_list)} specification(s)...')
        
        # Order specifications by criticality if multiple specs
        if len(specs_list) > 1:
            # Compute sensitivity
            S, _ = nn.calcSensitivity(x, options, store_sensitivity=False)
            
            # Import pagemtimes for batch matrix multiplication
            from cora_python.nn.layers.linear.nnGeneratorReductionLayer import pagemtimes
            
            # Compute criticality for each spec
            cv_list = []
            for spec_i in specs_list:
                A, _, is_safe = spec_to_linear_constraint(spec_i)
                
                # MATLAB: pagemtimes(-A, S) or pagemtimes(A, S)
                # S shape is typically (nK, output_dim, batch_size) or similar
                # A shape is (p, output_dim) where p is number of constraints
                if is_safe:
                    # MATLAB: grad = pagemtimes(-A, S)
                    grad = pagemtimes(-A, None, S, None)
                else:
                    # MATLAB: grad = pagemtimes(A, S)
                    grad = pagemtimes(A, None, S, None)
                
                # MATLAB: [p,~] = size(A); sgrad = reshape(permute(sign(grad),[2 3 1]),size(x).*[1 p])
                p = A.shape[0]
                # sign(grad) and permute to match MATLAB
                sgrad_sign = np.sign(grad)
                # Handle different grad shapes
                if grad.ndim == 3:
                    # Permute: [2 3 1] means swap dimensions
                    sgrad_permuted = np.transpose(sgrad_sign, (1, 2, 0))
                    # Reshape to match MATLAB: size(x).*[1 p]
                    sgrad = sgrad_permuted.reshape(x.shape[0], p)
                else:
                    # Fallback for 2D case
                    sgrad = sgrad_sign.T if sgrad_sign.ndim == 2 else sgrad_sign.flatten()[:x.shape[0]*p].reshape(x.shape[0], p)
                
                # MATLAB: x_ = repelem(x,1,p) + repelem(r,1,p).*sgrad
                from cora_python.nn.layers.linear.nnGeneratorReductionLayer import repelem
                x_repeated = repelem(x, 1, p)
                r_repeated = repelem(r, 1, p)
                xi_ = x_repeated + r_repeated * sgrad
                
                # Evaluate outputs
                yi_ = nn.evaluate(xi_, options)
                
                # Compute criticality (MATLAB: min(aux_computeCriticallityOfSpecs(specs(i),y_)))
                cv = compute_criticality_of_specs(spec_i, yi_)
                # MATLAB uses min over all values
                if isinstance(cv, np.ndarray):
                    cv = np.min(cv)
                cv_list.append(cv)
            
            # Sort specs by criticality
            # MATLAB: order = 'ascend' (hardest to easiest) unless verify_cascade_unsafe_set_constraints
            cascade_unsafe = options.get('nn', {}).get('verify_cascade_unsafe_set_constraints', False)
            if cascade_unsafe:
                # Order from easiest to hardest
                spec_order = np.argsort(cv_list)[::-1]  # descending
            else:
                # Order from hardest to easiest
                spec_order = np.argsort(cv_list)  # ascending
            specs_list = [specs_list[i] for i in spec_order]
        
        # Track unknown specifications
        there_is_unknown = False
        
        # MATLAB: Keep track of verified constraints for cascading
        A_verified = np.array([]).reshape(0, 0)  # Empty 2D array
        b_verified = np.array([]).reshape(0, 1)  # Empty column vector
        
        # Handle multiple specs
        for i, spec_i in enumerate(specs_list):
            A_i, b_i, is_safe_set = spec_to_linear_constraint(spec_i)
            
            # MATLAB: Add verified constraints if cascading is enabled
            cascade_unsafe = options.get('nn', {}).get('verify_cascade_unsafe_set_constraints', False)
            if cascade_unsafe and A_verified.size > 0:
                # Append verified constraints
                A = np.vstack([A_i, A_verified])
                b = np.vstack([b_i, b_verified])
            else:
                A = A_i
                b = b_i
            
            # MATLAB: numUnionConstraints = double(safeSet)*size(Ai,1)
            # In Python verify, safeSet is boolean, but we need to pass numUnionConstraints
            # For now, pass is_safe_set as boolean (Python verify will handle it)
            num_union_constraints = int(is_safe_set) * A_i.shape[0]
            
            try:
                # Compute remaining timeout
                rem_timeout = timeout - (time.time() - total_time_start)
                
                # Run verification
                # MATLAB: [res,x_,y_] = nn.verify(x,r,A,b,numUnionConstraints, options,remTimeout,verbose,[],false)
                # Python verify signature: verify(nn, x, r, A, b, safeSet, options, timeout, verbose, plotDims, plotSplittingTree)
                # For now, pass is_safe_set as boolean
                result_str_i, x_, y_ = nn.verify(x, r, A, b, is_safe_set, options, rem_timeout, verbose)
                
                # Note: Python version doesn't track numVerified currently
                # result['numVerified'] += 1  # Would need to be added to verify method
                
                result['str'] = result_str_i
                result['x'] = x_
                result['y'] = y_
                
            except Exception as e:
                if verbose:
                    print(f'\nUnexpected Error during verification: {e}')
                result['str'] = 'UNKNOWN'
                break
            
            if verbose:
                print(f' done (specification {i+1}/{len(specs_list)})')
            
            # Check result
            if result['str'] == 'VERIFIED':
                # MATLAB: Grow verified specification if safe set or single unsafe constraint
                if is_safe_set:
                    # We can grow the verified specification for safe sets
                    if A_verified.size > 0:
                        A_verified = np.vstack([A_verified, A_i])
                        b_verified = np.vstack([b_verified, b_i])
                    else:
                        A_verified = A_i
                        b_verified = b_i
                elif A_i.shape[0] == 1:
                    # Single unsafe constraint: add its inverse
                    if A_verified.size > 0:
                        A_verified = np.vstack([A_verified, -A_i])
                        b_verified = np.vstack([b_verified, -b_i])
                    else:
                        A_verified = -A_i
                        b_verified = -b_i
                # Continue with next specification
                continue
            elif result['str'] == 'COUNTEREXAMPLE':
                # Found counterexample, no need to check remaining specs
                there_is_unknown = False
                break
            else:
                # Could not verify this specification
                there_is_unknown = True
        
        if verbose:
            print('Writing results...')
        
        # If any spec is unknown, overall result is unknown
        if there_is_unknown:
            result['str'] = 'UNKNOWN'
        
        # Write results to file (MATLAB format: uses newline character)
        with open(results_path, 'w', newline='') as fid:
            if result['str'] == 'VERIFIED':
                result_str = 'unsat'
                fid.write('unsat\n')
            elif result['str'] == 'COUNTEREXAMPLE':
                result_str = 'sat'
                fid.write('sat\n(\n')
                # Write input values
                x_ = result['x'].flatten()
                # Reorder input dimensions if permute_dims was used (MATLAB: reshape(permute(reshape(x_,inSize([2 1 3])),[2 1 3]),[],1))
                if permute_dims and hasattr(nn, 'layers') and len(nn.layers) > 0:
                    in_size = nn.layers[0].inputSize
                    if len(in_size) == 3:
                        # Reshape to [W, H, C], permute to [H, W, C], then flatten
                        x_reshaped = x_.reshape((in_size[1], in_size[0], in_size[2]))
                        x_permuted = np.transpose(x_reshaped, (1, 0, 2))
                        x_ = x_permuted.flatten()
                # MATLAB: fprintf(fid,['(X_%d %f)' newline],j-1,x_(j))
                for j in range(len(x_)):
                    fid.write(f'(X_{j} {x_[j]:.6f})\n')
                # Write output values
                # MATLAB: fprintf(fid,['(Y_%d %f)' newline],j-1,y_(j))
                y_ = result['y'].flatten()
                for j in range(len(y_)):
                    fid.write(f'(Y_{j} {y_[j]:.6f})\n')
                fid.write(')')
            else:
                result_str = 'unknown'
                fid.write('unknown\n')
        
        if verbose:
            print(' done')
        
    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        result_str = 'unknown'
        with open(results_path, 'w') as fid:
            fid.write('unknown\n')
    finally:
        # Clean up temporary decompressed files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                if verbose:
                    print(f'Warning: Could not delete temporary file {temp_file}: {e}')
    
    if verbose:
        total_time = time.time() - total_time_start
        result['time'] = total_time
        print(f'{original_model_path} -- {original_vnnlib_path}: {result_str.upper()}')
        print(f'--- Verification time: {total_time:.4f} / {timeout:.4f} [s]')
    
    return result_str, result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VNN-COMP verification instance')
    parser.add_argument('bench_name', type=str, help='Benchmark name')
    parser.add_argument('model_path', type=str, help='Path to .onnx file')
    parser.add_argument('vnnlib_path', type=str, help='Path to .vnnlib file')
    parser.add_argument('results_path', type=str, help='Path to results file')
    parser.add_argument('timeout', type=float, help='Timeout in seconds')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    result_str, result_dict = run_instance(
        args.bench_name,
        args.model_path,
        args.vnnlib_path,
        args.results_path,
        args.timeout,
        args.verbose
    )
    
    print(f'Result: {result_str}')
    print(f'Verification result: {result_str.upper()}')

