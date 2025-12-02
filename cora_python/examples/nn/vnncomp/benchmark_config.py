"""
Benchmark-specific configuration for VNN-COMP benchmarks

This module contains tuned parameters for each VNN-COMP benchmark.
Configurations are extracted from prepare_instance.m

Authors:       Lukas Koller
Written:       11-August-2025
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import math
from typing import Dict, Any, Optional


def get_default_options() -> Dict[str, Any]:
    """
    Get default evaluation options for all benchmarks.
    
    Returns:
        Dictionary with default options
    """
    return {
        'nn': {
            'use_approx_error': True,
            'poly_method': 'bounds',  # {'bounds', 'singh', 'center'}
            'train': {
                'use_gpu': False,
                'backprop': False,
                'mini_batch_size': 1024  # 2^10
            },
            'interval_center': False,
            'batch_norm_moving_stats': True,
            'falsification_method': 'zonotack',  # {'center', 'fgsm', 'zonotack'}
            'refinement_method': 'zonotack',  # {'naive', 'zonotack', 'zonotack-layerwise'}
            'train.num_init_gens': math.inf,
            'approx_error_order': 'length',  # or 'sensitivity*'
            'exact_conzonotope_bounds': False,
            'num_splits': 2,
            'num_dimensions': 1,
            'num_neuron_splits': 0,
            'num_relu_constraints': 0,
            'add_orth_neuron_splits': False,
            'input_xor_neuron_splitting': True,
            'polytope_bound_approx_max_iter': 3,
            'refinement_max_iter': 3,
        }
    }


# Benchmark-specific configurations
BENCHMARK_CONFIGS = {
    'test': {
        'permute_dims': False,
        'input_data_formats': '',
        'output_data_formats': '',
        'target_network': 'dlnetwork',
        'contains_composite_layers': False,
        # Uses all default options
    },
    
    'acasxu_2023': {
        'permute_dims': False,
        'input_data_formats': 'BSSC',
        'output_data_formats': '',
        'options': {
            'nn': {
                'num_splits': 2,
                'num_dimensions': 1,
                # 'num_neuron_splits': 1,  # Optional
                # 'num_relu_constraints': math.inf,  # Optional
            }
        }
    },
    
    'cifar100': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'target_network': 'dagnetwork',
        'contains_composite_layers': True,
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': 1000,
                'train.num_approx_err': 100,
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 1,
                'train.mini_batch_size': 4,  # 2^2
                'batch_union_conzonotope_bounds': False,
                'max_verif_iter': 10,
            }
        }
    },
    
    'collins_rul_cnn_2023': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': math.inf,
                'train.num_approx_err': 100,
            }
        }
    },
    
    'collins_rul_cnn': {  # Alias
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': math.inf,
                'train.num_approx_err': 100,
            }
        }
    },
    
    'cora': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        'options': {
            'nn': {
                'num_relu_constraints': math.inf,
                'train.mini_batch_size': 32,  # 2^5
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 1,
                'batch_union_conzonotope_bounds': False,
            }
        }
    },
    
    'dist_shift_2023': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        # Uses default values
    },
    
    'linearizenn': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        'target_network': 'dagnetwork',
        'contains_composite_layers': True,
        'options': {
            'nn': {
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 0,
            }
        }
    },
    
    'metaroom_2023': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'options': {
            'nn': {
                'train.num_init_gens': 500,
                'train.num_approx_err': 100,
                'train.mini_batch_size': 4,  # 2^2
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 0,
                'num_relu_constraints': 0,
                'batch_union_conzonotope_bounds': False,
                'max_verif_iter': 10,
            }
        }
    },
    
    'nn4sys_2023': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': 'BC',
        'supported_models': ['lindex', 'lindex_deep'],
        # Uses default parameters
    },
    
    'safenlp': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        'options': {
            'nn': {
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 1,
                'num_relu_constraints': 100,
            }
        }
    },
    
    'safenlp_2024': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        'options': {
            'nn': {
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 1,
                'num_relu_constraints': 100,
            }
        }
    },
    
    'tinyimagenet': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'target_network': 'dagnetwork',
        'contains_composite_layers': True,
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': 500,
                'train.num_approx_err': 0,
                'num_relu_constraints': 0,
                'num_splits': 2,
                'num_dimensions': 1,
                'num_neuron_splits': 1,
                'batch_union_conzonotope_bounds': False,
                'train.mini_batch_size': 32,  # 2^5
            }
        }
    },
    
    'tllverifybench_2023': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        # Uses default parameters
    },
    
    # VNN-COMP'22 Benchmarks
    'mnist_fc': {
        'permute_dims': False,
        'input_data_formats': 'SSC',
        'output_data_formats': '',
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': 500,
                'train.num_approx_err': 100,
                'batch_union_conzonotope_bounds': False,
            }
        }
    },
    
    'oval21': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        'options': {
            'nn': {
                'interval_center': True,
                'train.num_init_gens': 100,
                'train.num_approx_err': 10,
                'batch_union_conzonotope_bounds': False,
            }
        }
    },
    
    'reach_prob_density': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        # Uses default parameters
    },
    
    'rl_benchmarks': {
        'permute_dims': False,
        'input_data_formats': 'BC',
        'output_data_formats': '',
        'unsupported_specs': [
            'vnnlib/dubinsrejoin_case_safe_10.vnnlib',
            'vnnlib/dubinsrejoin_case_safe_13.vnnlib',
            'vnnlib/dubinsrejoin_case_safe_15.vnnlib',
            'vnnlib/dubinsrejoin_case_safe_16.vnnlib',
            'vnnlib/dubinsrejoin_case_safe_17.vnnlib',
        ],
        # Uses default parameters
    },
    
    'sri_resnet_a': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        # Uses default parameters
    },
    
    'sri_resnet_b': {
        'permute_dims': True,
        'input_data_formats': 'BCSS',
        'output_data_formats': '',
        # Uses default parameters
    },
}

# Unsupported benchmarks
UNSUPPORTED_BENCHMARKS = [
    'cctsdb_yolo_2023',
    'cgan_2023',
    'collins_aerospace_benchmark',
    'collins_yolo_robustness_2023',
    'lsnc',
    'ml4acopf_2023',
    'ml4acopf_2024',
    'traffic_signs_recognition_2023',
    'vggnet16_2023',
    'vit_2023',
    'yolo_2023',
]


def get_benchmark_options(bench_name: str, model_name: Optional[str] = None,
                          vnnlib_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get options for a specific benchmark.
    
    Args:
        bench_name: Name of the benchmark
        model_name: Name of the model (optional, for model-specific configs)
        vnnlib_path: Path to vnnlib file (optional, for spec-specific configs)
        
    Returns:
        Dictionary with benchmark options
        
    Raises:
        ValueError: If benchmark is unsupported
    """
    # Handle aliases (e.g., safenlp_2024 -> safenlp)
    if bench_name == 'safenlp_2024':
        bench_name = 'safenlp'
    
    # Check if benchmark is unsupported
    if bench_name in UNSUPPORTED_BENCHMARKS:
        raise ValueError(f"Benchmark '{bench_name}' is not supported")
    
    # Check for unsupported models in nn4sys
    if bench_name == 'nn4sys_2023' and model_name is not None:
        supported_models = BENCHMARK_CONFIGS[bench_name].get('supported_models', [])
        if model_name not in supported_models:
            raise ValueError(f"Model '{model_name}' of benchmark '{bench_name}' is not supported")
    
    # Check for unsupported specs in rl_benchmarks
    if bench_name == 'rl_benchmarks' and vnnlib_path is not None:
        unsupported_specs = BENCHMARK_CONFIGS[bench_name].get('unsupported_specs', [])
        if vnnlib_path in unsupported_specs:
            raise ValueError(f"Specification '{vnnlib_path}' of benchmark '{bench_name}' is not supported")
    
    # Get benchmark config or use defaults
    bench_config = BENCHMARK_CONFIGS.get(bench_name, {})
    
    # Start with default options
    options = get_default_options()
    
    # Merge benchmark-specific options
    if 'options' in bench_config:
        bench_options = bench_config['options']
        if 'nn' in bench_options:
            # Deep merge nn options
            for key, value in bench_options['nn'].items():
                if '.' in key:
                    # Handle nested keys like 'train.num_init_gens'
                    parts = key.split('.')
                    target = options['nn']
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    options['nn'][key] = value
    
    return {
        'options': options,
        'permute_dims': bench_config.get('permute_dims', False),
        'input_data_formats': bench_config.get('input_data_formats', ''),
        'output_data_formats': bench_config.get('output_data_formats', ''),
        'target_network': bench_config.get('target_network', 'dlnetwork'),
        'contains_composite_layers': bench_config.get('contains_composite_layers', False),
    }


if __name__ == '__main__':
    # Test configuration retrieval
    print("Testing benchmark configurations...")
    
    for bench_name in ['test', 'acasxu_2023', 'cifar100', 'cora']:
        try:
            config = get_benchmark_options(bench_name)
            print(f"\n{bench_name}:")
            print(f"  Permute dims: {config['permute_dims']}")
            print(f"  Input formats: {config['input_data_formats']}")
            print(f"  Batch size: {config['options']['nn']['train']['mini_batch_size']}")
            print(f"  Num splits: {config['options']['nn']['num_splits']}")
        except Exception as e:
            print(f"\n{bench_name}: Error - {e}")
    
    # Test unsupported benchmark
    try:
        get_benchmark_options('cgan_2023')
        print("\nERROR: Should have raised exception for unsupported benchmark")
    except ValueError as e:
        print(f"\nCorrectly caught unsupported benchmark: {e}")

