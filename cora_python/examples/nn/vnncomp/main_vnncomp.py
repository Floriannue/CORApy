"""
main_vnncomp - run all scripts to repeat the evaluation

Results will be saved to ./results

Syntax:
    completed = main_vnncomp()
    completed = main_vnncomp(eval_name)

Inputs:
    eval_name - str, results are stored in ./results/<eval_name>
                defaults to 'cora'

Outputs:
    completed - boolean

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import sys
import os
import time
from datetime import datetime
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from scripts.run_benchmarks import run_benchmarks


def main_vnncomp(eval_name: Optional[str] = None) -> bool:
    """
    Main evaluation script for VNN-COMP.
    
    Args:
        eval_name: Name for this evaluation run (default: 'cora')
        
    Returns:
        Boolean indicating success
    """
    # 1. SETTINGS -------------------------------------------------------------
    
    PAPER_TITLE = 'CORA Python'
    VENUE_NAME = 'VNN-COMP'
    
    run_startup(PAPER_TITLE, VENUE_NAME)
    
    # 2. SETUP ----------------------------------------------------------------
    
    # Parse input
    if eval_name is None:
        eval_name = 'cora'
    
    # Set up paths
    base_path = '.'
    
    code_path, data_path, results_path = setup_paths(base_path, eval_name)
    
    # 3. RUN SCRIPTS ----------------------------------------------------------
    
    benchmarks = [
        'test',
        # VNN-COMP'25 benchmarks (uncomment as needed)
        # 'acasxu_2023',
        # 'cctsdb_yolo_2023',  # (not supported; not main track)
        # 'cersyve',  # (test)
        # 'cgan_2023',  # (not supported)
        # 'cifar100_2024',  # (test & fix)
        # 'collins_aerospace_benchmark',  # (not supported)
        # 'collins_rul_cnn_2022',
        # 'cora_2024',
        # 'dist_shift_2023',
        # 'linearizenn_2024',
        # 'lsnc_relu',  # (not supported)
        # 'malbeware',
        # 'metaroom_2023',
        # 'ml4acopf_2024',  # (not supported)
        # 'sat_relu',
        # 'nn4sys',  # (not supported; TODO: missing convolution 1D)
        # 'relusplitter',  # (test)
        # 'safenlp_2024',
        # 'soundnessbench',  # (TODO: tune parameters)
        # 'tinyimagenet_2024',
        # 'tllverifybench_2023',
        # 'traffic_signs_recognition_2023',  # (not supported; not main track)
        # 'vggnet16_2022',  # (not supported; not main track)
        # 'vit_2023',  # (not supported)
        # 'yolo_2023',  # (not supported)
    ]
    
    scripts = [
        (lambda: run_benchmarks(benchmarks, data_path, results_path), 'run_benchmarks'),
    ]
    
    # Run scripts
    run_scripts(scripts)
    
    # 4. WRAP UP --------------------------------------------------------------
    
    wrap_up()
    
    return True


# Auxiliary functions ---------------------------------------------------------

def run_startup(paper_title: str, venue_name: str):
    """Show startup block."""
    print()
    print('='*70)
    print()
    print('Repeatability Package')
    print(f'Paper: {paper_title}')
    print(f'Venue: {venue_name}')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # Try to get CORA version
    try:
        import cora_python
        if hasattr(cora_python, '__version__'):
            print(f'CORA Python: {cora_python.__version__}')
    except:
        print('CORA Python: version unknown')
    
    print(f'Python: {sys.version.split()[0]}')
    print(f'System: {sys.platform}')
    print()
    print('='*70)
    print()
    time.sleep(1)  # Pause to make startup block readable


def setup_paths(base_path: str, eval_name: str):
    """
    Set up paths for evaluation.
    
    Returns:
        Tuple of (code_path, data_path, results_path)
    """
    # Set up paths
    code_path = os.path.join(base_path, 'code')
    data_path = os.path.join(base_path, 'data')
    
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    results_path = os.path.join(base_path, 'results', timestamp, eval_name)
    
    # Create results directory
    os.makedirs(results_path, exist_ok=True)
    
    # Set up logging to file
    results_txt = os.path.join(results_path, 'results.txt')
    
    # Redirect output to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Note: This is simplified - full dual output would need more setup
    print(f'Results will be saved to: {results_path}')
    print(f'Log file: {results_txt}')
    
    return code_path, data_path, results_path


def run_scripts(scripts):
    """
    Run all scripts in the list.
    
    Args:
        scripts: List of (function, name) tuples
    """
    n = len(scripts)
    print(f"Running {n} script(s)...")
    print()
    
    for i, (script_func, name) in enumerate(scripts):
        # Run script i
        print('-'*70)
        print()
        
        try:
            # Call script
            print(f"Running '{name}' ...")
            script_func()
            print()
            print(f"'{name}' was run successfully!")
        except Exception as e:
            # Error handling
            print()
            print(f"An ERROR occurred during execution of '{name}':")
            print(str(e))
            import traceback
            traceback.print_exc()
            print("Continuing with next script...")
        
        print()


def wrap_up():
    """Wrap up evaluation."""
    print('='*70)
    print()
    print("Completed!")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VNN-COMP evaluation')
    parser.add_argument('--eval-name', type=str, default='cora',
                       help='Name for this evaluation run')
    
    args = parser.parse_args()
    
    completed = main_vnncomp(args.eval_name)
    
    if completed:
        print('Evaluation completed successfully!')
        sys.exit(0)
    else:
        print('Evaluation failed!')
        sys.exit(1)

