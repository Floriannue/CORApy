"""
vnnlib2cora - import specifications from .vnnlib files (MATLAB-faithful translation)

Description:
    Import specifications from .vnnlib files and convert them to CORA format

Syntax:
    [X0,spec] = vnnlib2cora(file)

Inputs:
    file - path to a file .vnnlib file storing the specification

Outputs:
    X0 - initial set represented as an object of class interval
    spec - specifications represented as an object of class specification

Reference:
    - https://www.vnnlib.org/

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2021
Last update:   26-July-2023 (TL, speed up)
                30-August-2023 (TL, bug fix multiple terms in and)
                14-June-2024 (TL, major speed up)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import re
from typing import Tuple, List, Dict, Any, Union
import numpy as np

# Import CORA Python modules
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope
from cora_python.specification.specification import Specification


def vnnlib2cora(file_path: str) -> Tuple[List[Interval], Specification]:
    """
    Import specifications from .vnnlib files and convert them to CORA format.
    
    Args:
        file_path: Path to the .vnnlib file
        
    Returns:
        Tuple of (X0, spec) where:
            X0: List of initial sets as Interval objects
            spec: Specification object
    """
    # read in text from file
    with open(file_path, 'r') as f:
        text = f.read()
    
    # determine number of inputs and number of outputs
    nrInputs = 0
    nrOutputs = 0
    lineBreaks = [m.start() for m in re.finditer(r'\n', text)]
    
    for ln in range(len(lineBreaks) - 1):
        # iterate through file
        i = lineBreaks[ln] + 1
        i1 = lineBreaks[ln + 1]
        if text[i:i1].strip().startswith('(declare-const '):
            temp = text[i + 15:i1].strip()
            ind = temp.find(' ')
            if ind != -1:
                temp = temp[:ind]
                if temp[0] == 'X':
                    # found new input; read out index
                    nrInputs = max(nrInputs, int(temp[2:]))
                elif temp[0] == 'Y':
                    # found new output; read out index
                    nrOutputs = max(nrOutputs, int(temp[2:]))
    
    # +1 due to 0-indexing in vnnlib files
    data = {
        'nrInputs': nrInputs + 1,
        'nrOutputs': nrOutputs + 1,
        'currIn': 0,
        'polyInput': [],
        'polyOutput': []
    }
    
    # parse file
    while text.strip():
        if text.strip().startswith('(assert'):
            text = text.strip()[7:].strip()  # Remove '(assert' and trim
            len_parsed, data = aux_parseAssert(text, data)
            text = text.strip()[len_parsed:].strip()
        else:
            ln = text.find('\n')
            if ln != -1:
                text = text[ln + 1:]
            else:
                break
    
    # convert data to polytopes ---
    
    # a) convert input
    # potentially convert input polytopes to intervals
    X0 = []
    for i in range(len(data['polyInput'])):
        polyStruct = data['polyInput'][i]
        P = Polytope(polyStruct['C'], polyStruct['d'])
        
        # Check if polytope can be represented as interval
        try:
            # Try to convert to interval representation
            I = P.interval()
            X0.append(I)
        except:
            raise ValueError('Input set is not an interval.')
    
    # b) convert output
    Y = []
    for i in range(len(data['polyOutput'])):
        Y.append(Polytope(data['polyOutput'][i]['C'], data['polyOutput'][i]['d']))
    
    # construct specification from list of output polytopes
    if not Y:
        raise ValueError(f"Unable to convert file: {file_path}")
    elif len(Y) == 1:
        # MATLAB: spec = specification(Y{1}, 'safeSet');
        spec = Specification(Y[0], 'safeSet')
    else:
        # MATLAB: Y = safeSet2unsafeSet(Y); spec = []; for i = 1:length(Y) spec = add(spec, specification(Y{i}, 'unsafeSet')); end
        Y_unsafe = safeSet2unsafeSet(Y)
        spec = None
        for i in range(len(Y_unsafe)):
            spec_i = Specification(Y_unsafe[i], 'unsafeSet')
            if spec is None:
                spec = spec_i
            else:
                spec = spec.add(spec_i)
    
    # vnnlib files have specifications inverted
    spec = spec.inverse()
    
    return X0, spec


def safeSet2unsafeSet(S: List[Polytope]) -> List[Polytope]:
    """
    Convert the union of safe sets to an equivalent representation as the union of unsafe sets.
    
    Args:
        S: List of safe sets (Polytope objects)
        
    Returns:
        List of equivalent unsafe sets represented as Polytope objects
    """
    # represent first safe set by the union of unsafe sets
    F = aux_getUnsafeSets(S[0])
    nrTotalSets = len(F)
    
    # loop over all safe sets
    for i in range(1, len(S)):
        # represent current safe set by the union of unsafe sets
        F_i = aux_getUnsafeSets(S[i])
        nrAddSets = len(F_i)
        
        # compute the intersection with the previous unsafe sets
        F_ = []
        for j in range(nrTotalSets):
            for k in range(nrAddSets):
                # Compute intersection F[j] & F_i[k]
                try:
                    intersection = F[j].and_(F_i[k])
                    if not intersection.isemptyobject():
                        F_.append(intersection)
                except:
                    # If intersection fails, skip
                    pass
        
        # Update F with non-empty polytopes
        F = F_
        nrTotalSets = len(F)
    
    return F


def aux_getUnsafeSets(S: Polytope) -> List[Polytope]:
    """
    Represent the safe set S as a union of unsafe sets.
    
    Args:
        S: Safe set as Polytope
        
    Returns:
        List of unsafe sets
    """
    # convert to polytope (already is one)
    P = S
    
    # loop over all polytope halfspaces and invert them
    nrCon = P.A.shape[0]
    F = []
    
    for i in range(nrCon):
        # Create complement of halfspace P.A[i,:] * x <= P.b[i]
        # Complement is P.A[i,:] * x > P.b[i], which is -P.A[i,:] * x < -P.b[i]
        A_neg = -P.A[i:i+1, :]  # Keep as 2D array
        b_neg = -P.b[i:i+1]     # Keep as 2D array
        F.append(Polytope(A_neg, b_neg))
    
    return F


# Auxiliary functions (matching MATLAB exactly) -----------------------------------------------------

def aux_parseAssert(text: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Parse one assert statement."""
    
    if text.startswith('(<=') or text.startswith('(>='):
        return aux_parseLinearConstraint(text, data)
    
    elif text.startswith('(or'):
        text = text[3:].strip()
        data_ = {
            'spec': [],
            'nrOutputs': data['nrOutputs'],
            'nrInputs': data['nrInputs'],
            'polyInput': [],
            'polyOutput': [],
            'currIn': 0
        }
        len_total = 4  # for '(or '
        
        # parse all or conditions
        while not text.startswith(')'):
            # parse one or condition
            len_parsed, data_ = aux_parseAssert(text, data_)
            
            # update remaining text
            text = text[len_parsed:].strip()
            len_total += len_parsed
            
            # update input conditions
            if data_['polyInput']:
                if data['polyInput']:
                    data['polyInput'].append(data_['polyInput'][0])
                else:
                    data['polyInput'] = data_['polyInput']
            
            # update output conditions
            if data_['polyOutput']:
                if data['polyOutput']:
                    data['polyOutput'].append(data_['polyOutput'][0])
                else:
                    data['polyOutput'] = data_['polyOutput']
        
        return len_total + 1, data  # +1 for closing ')'
    
    elif text.startswith('(and'):
        text = text[4:].strip()
        len_total = 5  # for '(and '
        
        # parse all and conditions
        while not text.startswith(')'):
            len_parsed, data = aux_parseAssert(text, data)
            text = text[len_parsed:]
            
            # trim white spaces
            text_stripped = text.strip()
            len_total += len_parsed + (len(text) - len(text_stripped))
            text = text_stripped
            
            if text.startswith('('):
                # multiple terms in and; correct len
                len_total -= 1
        
        return len_total + 1, data  # +1 for closing ')'
    
    else:
        raise ValueError(f'Failed to parse vnnlib file. Unknown assert format: {text[:50]}')


def aux_createPolytopeStruct(n: int) -> Dict[str, np.ndarray]:
    """Create empty polytope structure."""
    return {
        'C': np.zeros((2*n, n)),
        'd': np.zeros((2*n, 1))
    }


def aux_parseLinearConstraint(text: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Parse a linear constraint."""
    
    # extract operator
    op = text[1:3]  # '<=' or '>='
    text = text[4:].strip()
    len_total = 4
    
    # get type of constraint (on inputs X or on output Y)
    constraint_type = aux_getTypeOfConstraint(text)
    
    # initialization
    if constraint_type == 'input':
        C = np.zeros((1, data['nrInputs']))
        d = 0.0
    else:
        C = np.zeros((1, data['nrOutputs']))
        d = 0.0
    
    # parse first argument
    C1, d1, len_parsed = aux_parseArgument(text, C.copy(), d)
    len_total += len_parsed
    text = text[len_parsed:].strip()
    
    # parse second argument
    C2, d2, len_parsed = aux_parseArgument(text, C.copy(), d)
    len_total += len_parsed
    
    # combine the two arguments
    if op == '<=':
        C = C1 - C2
        d = d2 - d1
    else:  # op == '>='
        C = C2 - C1
        d = d1 - d2
    
    # combine the current constraint with previous constraints
    if constraint_type == 'input':
        if not data['polyInput']:
            data['polyInput'] = [aux_createPolytopeStruct(data['nrInputs'])]
        
        data['currIn'] += 1
        
        for i in range(len(data['polyInput'])):
            data['polyInput'][i]['C'][data['currIn']-1, :] = C
            data['polyInput'][i]['d'][data['currIn']-1] = d
    
    else:  # output
        if not data['polyOutput']:
            data['polyOutput'] = [aux_createPolytopeStruct(0)]
        
        for i in range(len(data['polyOutput'])):
            # Append to existing constraints
            current_C = data['polyOutput'][i]['C']
            current_d = data['polyOutput'][i]['d']
            
            # Resize arrays if needed
            if current_C.shape[1] == 0:
                data['polyOutput'][i]['C'] = C
                data['polyOutput'][i]['d'] = np.array([[d]])
            else:
                data['polyOutput'][i]['C'] = np.vstack([current_C, C])
                data['polyOutput'][i]['d'] = np.vstack([current_d, [[d]]])
    
    return len_total, data


def aux_parseArgument(text: str, C: np.ndarray, d: float) -> Tuple[np.ndarray, float, int]:
    """Parse next argument."""
    
    if text.startswith('X') or text.startswith('Y'):
        # Find end of variable name
        end_idx = 1
        for i in range(1, len(text)):
            if text[i] in [' ', ')']:
                end_idx = i
                break
        
        index = int(text[2:end_idx])  # Convert X_0 -> 0, Y_1 -> 1, etc.
        C[0, index] += 1.0
        
        return C, d, end_idx
    
    elif text.startswith('(+'):
        # parse first argument
        C1, d1, len1 = aux_parseArgument(text[2:].strip(), C.copy(), d)
        text = text[len1+2:].strip()
        
        # parse second argument
        C2, d2, len2 = aux_parseArgument(text, C.copy(), d)
        
        # combine both arguments
        return C1 + C2, d1 + d2, len1 + len2 + 2
    
    elif text.startswith('(-'):
        # parse first argument
        C1, d1, len1 = aux_parseArgument(text[2:].strip(), C.copy(), d)
        text = text[len1+2:].strip()
        
        # parse second argument
        C2, d2, len2 = aux_parseArgument(text, C.copy(), d)
        
        # combine both arguments
        return C1 - C2, d1 - d2, len1 + len2 + 2
    
    else:
        # Parse numerical constant
        end_idx = 1
        for i in range(1, len(text)):
            if text[i] in [' ', ')']:
                end_idx = i
                break
        
        value = float(text[:end_idx])
        return C, d + value, end_idx


def aux_getTypeOfConstraint(text: str) -> str:
    """Check if the current constraint is on the inputs or on the outputs."""
    
    indX = text.find('X')
    indY = text.find('Y')
    
    # either X or Y must be given
    if indX == -1:
        if indY == -1:
            # none given
            raise ValueError('File format not supported')
        else:
            # Y is not empty
            return 'output'
    elif indY == -1:
        # X is not empty
        return 'input'
    else:
        # return smaller
        if indX < indY:
            return 'input'
        else:
            return 'output'
