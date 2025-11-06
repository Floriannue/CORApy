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
    
    try:
        # Measure verification time
        total_time_start = time.time()
        
        if verbose:
            print('--- Loading network...')
        
        # Load neural network
        nn = NeuralNetwork.readONNXNetwork(model_path)
        
        if verbose:
            print(' done')
            print('--- Loading specification...')
        
        # Load specification
        X0, specs = vnnlib2cora(vnnlib_path)
        
        if verbose:
            print(' done')
        
        # Get benchmark-specific options
        from get_instance_filename import get_instance_filename
        _, model_name, vnnlib_name = get_instance_filename(bench_name, model_path, vnnlib_path)
        
        try:
            bench_config = get_benchmark_options(bench_name, model_name, vnnlib_path)
            options = bench_config['options']
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
        
        options = validateNNoptions(options, True)
        
        # Batch all input sets
        x_list = []
        r_list = []
        for X0_i in X0:
            xi = (X0_i.sup + X0_i.inf) / 2
            ri = (X0_i.sup - X0_i.inf) / 2
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
            
            # Compute criticality for each spec
            cv_list = []
            for spec_i in specs_list:
                A, _, is_safe = spec_to_linear_constraint(spec_i)
                
                # Compute adversarial gradient
                if is_safe:
                    grad = -A.T @ S  # Shape depends on S dimensions
                else:
                    grad = A.T @ S
                
                # Compute adversarial attacks
                sgrad = np.sign(grad)
                xi_ = x + r * sgrad
                
                # Evaluate outputs
                yi_ = nn.evaluate(xi_, options)
                
                # Compute criticality
                cv = compute_criticality_of_specs(spec_i, yi_)
                cv_list.append(cv)
            
            # Sort specs by criticality (most critical first)
            spec_order = np.argsort(cv_list)
            specs_list = [specs_list[i] for i in spec_order]
        
        # Track unknown specifications
        there_is_unknown = False
        
        # Handle multiple specs
        for i, spec_i in enumerate(specs_list):
            A, b, is_safe_set = spec_to_linear_constraint(spec_i)
            
            try:
                # Compute remaining timeout
                rem_timeout = timeout - (time.time() - total_time_start)
                
                # Run verification
                # Python verify returns (result_str, x_, y_)
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
        
        # Write results to file
        with open(results_path, 'w') as fid:
            if result['str'] == 'VERIFIED':
                result_str = 'unsat'
                fid.write('unsat\n')
            elif result['str'] == 'COUNTEREXAMPLE':
                result_str = 'sat'
                fid.write('sat\n(\n')
                # Write input values
                x_ = result['x'].flatten()
                for j in range(len(x_)):
                    fid.write(f'(X_{j} {x_[j]:.6f})\n')
                # Write output values
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
    
    if verbose:
        total_time = time.time() - total_time_start
        result['time'] = total_time
        print(f'{model_path} -- {vnnlib_path}: {result_str.upper()}')
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

