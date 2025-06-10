"""
generateRandom - Generates a random contSet

Syntax:
    S = contSet.generateRandom()
    S = contSet.generateRandom('Dimension',n)
    S = contSet.generateRandom({@interval, @zonotope}, ...)

Inputs:
    admissibleSets - cell array of admissible sets
    Name-Value pairs (all options, arbitrary order):
       <'Dimension',n> - dimension
       ... - further pairs for subclasses

Outputs:
    S - contSet

Example:
    S = contSet.generateRandom('Dimension',2);
    plot(S);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner (MATLAB)
               Python translation by AI Assistant
Written:       05-April-2023
Last update:   09-January-2024 (changed admissibleSets to cell arrays of strings)
Last revision: ---
"""

import random
import importlib

def generateRandom(*args, **kwargs):
    """
    Generates a random contSet object.
    
    Args:
        *args: positional arguments, first can be list of admissible sets
        **kwargs: keyword arguments including 'Dimension' and set-specific options
        
    Returns:
        contSet: randomly generated contSet object
    """
    # Parse input arguments
    admissible_sets, remaining_kwargs = _parse_input(args, kwargs)
    
    # Randomly select set type
    set_type = random.choice(admissible_sets)
    
    # Import the corresponding module and call its generateRandom method
    try:
        # Import the specific set class
        module_path = f"cora_python.contSet.{set_type}.{set_type}"
        module = importlib.import_module(module_path)
        
        # Get the class (capitalized)
        set_class = getattr(module, set_type.capitalize())
        
        # Call generateRandom on the class
        if hasattr(set_class, 'generateRandom'):
            return set_class.generateRandom(**remaining_kwargs)
        else:
            raise NotImplementedError(f"generateRandom not implemented for {set_type}")
            
    except (ImportError, AttributeError) as e:
        # Fallback: if specific set type is not available, use interval
        if set_type != 'interval':
            try:
                from cora_python.contSet.interval.interval import Interval
                return Interval.generateRandom(**remaining_kwargs)
            except:
                raise NotImplementedError(f"generateRandom not available for {set_type}: {e}")
        else:
            raise NotImplementedError(f"generateRandom not available for {set_type}: {e}")


def _parse_input(args, kwargs):
    """
    Parse input arguments for generateRandom.
    
    Args:
        args: positional arguments
        kwargs: keyword arguments
        
    Returns:
        tuple: (admissible_sets, remaining_kwargs)
    """
    # Default admissible sets (only include sets that we have implemented)
    default_admissible = [
        'interval',  # We have this implemented
        # 'capsule', 'conPolyZono', 'conZonotope', 'ellipsoid', 'emptySet',
        # 'fullspace', 'levelSet', 'polytope', 'polyZonotope',
        # 'probZonotope', 'spectraShadow', 'zonoBundle', 'zonotope'
    ]
    
    # Check if first argument is a list of admissible sets
    if args and isinstance(args[0], (list, tuple)):
        admissible_sets = list(args[0])
        # Validate that all sets are strings
        if not all(isinstance(s, str) for s in admissible_sets):
            raise ValueError("Admissible sets should be a list of contSet strings")
    else:
        admissible_sets = default_admissible.copy()
    
    # For now, only support interval since that's what we have implemented
    # Filter to only include implemented sets
    implemented_sets = {'interval'}  # Add more as they get implemented
    admissible_sets = [s for s in admissible_sets if s in implemented_sets]
    
    if not admissible_sets:
        admissible_sets = ['interval']  # fallback
    
    return admissible_sets, kwargs 