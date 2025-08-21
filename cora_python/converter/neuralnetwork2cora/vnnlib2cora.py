"""
vnnlib2cora - import specifications from .vnnlib files

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

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: specification

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
        'currIn': 0
    }
    
    # parse file
    data['polyInput'] = []
    data['polyOutput'] = []
    while text.strip():
        if text.strip().startswith('(assert'):
            text = text.strip()[8:]  # Remove '(assert'
            len_parsed, data = aux_parseAssert(text, data)
            text = text.strip()[len_parsed + 1:]
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
        res, I = P.representsa_('interval', eps=1e-10)
        
        if res:
            X0.append(I)
        else:
            raise ValueError("Input set is not an interval.")
    
    # b) convert output
    Y = []
    for i in range(len(data['polyOutput'])):
        Y.append(Polytope(data['polyOutput'][i]['C'], data['polyOutput'][i]['d']))
    
    # construct specification from list of output polytopes
    if not Y:
        raise ValueError(f"Unable to convert file: {file_path}")
    elif len(Y) == 1:
        spec = Specification(Y[0], 'safeSet')
    else:
        # convert to the union of unsafe sets
        Y = safeSet2unsafeSet(Y)
        spec = None
        
        for i in range(len(Y)):
            if spec is None:
                spec = Specification(Y[i], 'unsafeSet')
            else:
                # Use the add function to combine specifications
                spec = spec.add(Specification(Y[i], 'unsafeSet'))
    
    # vnnlib files have specifications inverted
    spec = spec.inverse()
    
    return X0, spec


# Auxiliary functions -----------------------------------------------------

def aux_parseAssert(text: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Parse one assert statement.
    
    Args:
        text: Text to parse
        data: Data structure to update
        
    Returns:
        Tuple of (length_parsed, updated_data)
    """
    if text.strip().startswith('(<=') or text.strip().startswith('(>='):
        return aux_parseLinearConstraint(text, data)
    elif text.strip().startswith('(or'):
        text = text.strip()[4:]  # Remove '(or'
        data_ = {
            'spec': [],
            'nrOutputs': data['nrOutputs']
        }
        len_parsed = 5
        
        # parse all or conditions
        while not text.strip().startswith(')'):
            # parse one or condition
            data_['nrInputs'] = data['nrInputs']
            data_['nrOutputs'] = data['nrOutputs']
            data_['polyInput'] = []
            data_['polyOutput'] = []
            data_['currIn'] = 0
            
            len_, data_ = aux_parseAssert(text, data_)
            
            # update remaining text
            text = text.strip()[len_:]
            len_parsed += len_
            
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
        
        return len_parsed, data
    
    elif text.strip().startswith('(and'):
        text = text.strip()[5:]  # Remove '(and'
        len_parsed = 6
        
        # parse all and conditions
        while not text.strip().startswith(')'):
            len_, data = aux_parseAssert(text, data)
            text = text[len_:]
            
            # trim white spaces
            text_ = text.strip()
            len_ += (len(text) - len(text_))
            text = text_
            
            # move overall length counter to current position
            len_parsed += len_
            if text.startswith('('):
                # multiple terms in and; correct len
                len_parsed -= 1
        
        return len_parsed, data
    
    else:
        raise ValueError(f"Failed to parse vnnlib file. Parsed up to line {len_parsed}.")


def aux_createPolytopeStruct(n: int) -> Dict[str, np.ndarray]:
    """
    Create a polytope structure.
    
    Args:
        n: Number of dimensions
        
    Returns:
        Polytope structure dictionary
    """
    return {
        'C': np.zeros((2 * n, n)),
        'd': np.zeros((2 * n, 1))
    }


def aux_parseLinearConstraint(text: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Parse a linear constraint.
    
    Args:
        text: Text to parse
        data: Data structure to update
        
    Returns:
        Tuple of (length_parsed, updated_data)
    """
    # extract operator
    op = text[1:3]
    text = text[5:]
    len_parsed = 5
    
    # get type of constraint (on inputs X or on output Y)
    constraint_type = aux_getTypeOfConstraint(text)
    
    # initialization
    if constraint_type == 'input':
        C = np.zeros((1, data['nrInputs']))
        d = 0
    else:
        C = np.zeros((1, data['nrOutputs']))
        d = 0
    
    # parse first argument
    C1, d1, len_ = aux_parseArgument(text, C, d)
    len_parsed += len_
    text = text.strip()[len_:]
    
    # parse second argument
    C2, d2, len_ = aux_parseArgument(text, C, d)
    len_parsed += len_
    
    # combine the two arguments
    if op == '<=':
        C = C1 - C2
        d = d2 - d1
    else:
        C = C2 - C1
        d = d1 - d2
    
    # combine the current constraint with previous constraints
    if constraint_type == 'input':
        if not data['polyInput']:
            data['polyInput'] = [aux_createPolytopeStruct(data['nrInputs'])]
        
        data['currIn'] += 1
        
        for i in range(len(data['polyInput'])):
            data['polyInput'][i]['C'][data['currIn'] - 1, :] = C
            data['polyInput'][i]['d'][data['currIn'] - 1] = d
    
    else:  # output
        if not data['polyOutput']:
            data['polyOutput'] = [aux_createPolytopeStruct(0)]
        
        for i in range(len(data['polyOutput'])):
            data['polyOutput'][i]['C'] = np.vstack([data['polyOutput'][i]['C'], C])
            data['polyOutput'][i]['d'] = np.vstack([data['polyOutput'][i]['d'], d])
    
    return len_parsed, data


def aux_parseArgument(text: str, C: np.ndarray, d: float) -> Tuple[np.ndarray, float, int]:
    """
    Parse next argument.
    
    Args:
        text: Text to parse
        C: Coefficient matrix
        d: Constant term
        
    Returns:
        Tuple of (updated_C, updated_d, length_parsed)
    """
    if text.strip().startswith('X') or text.strip().startswith('Y'):
        # Find end of variable name
        len_parsed = 0
        for i in range(len(text)):
            if text[i] in ' )':
                len_parsed = i
                break
        
        index = int(text[2:len_parsed]) + 1
        C[0, index - 1] += 1
        return C, d, len_parsed
    
    elif text.strip().startswith('(+'):
        # parse first argument
        C1, d1, len_parsed = aux_parseArgument(text, C, d)
        text = text.strip()[len_parsed:]
        
        # parse second argument
        C2, d2, len_ = aux_parseArgument(text, C, d)
        len_parsed += len_
        
        # combine both arguments
        C = C1 + C2
        d = d1 + d2
        
        return C, d, len_parsed
    
    elif text.strip().startswith('(-'):
        # parse first argument
        C1, d1, len_parsed = aux_parseArgument(text, C, d)
        text = text.strip()[len_parsed:]
        
        # parse second argument
        C2, d2, len_ = aux_parseArgument(text, C, d)
        len_parsed += len_
        
        # combine both arguments
        C = C1 - C2
        d = d1 - d2
        
        return C, d, len_parsed
    
    else:
        # Find end of number
        len_parsed = 0
        for i in range(len(text)):
            if text[i] in ' )':
                len_parsed = i
                break
        
        d += float(text[:len_parsed])
        return C, d, len_parsed


def aux_getTypeOfConstraint(text: str) -> str:
    """
    Check if the current constraint is on the inputs or on the outputs.
    
    Args:
        text: Text to analyze
        
    Returns:
        Type of constraint ('input' or 'output')
    """
    indX = text.find('X')
    indY = text.find('Y')
    
    # either X or Y must be given
    if indX == -1:
        if indY == -1:
            # none given
            raise ValueError("File format not supported")
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


def safeSet2unsafeSet(safe_sets: List[Polytope]) -> List[Polytope]:
    """
    Convert safe sets to unsafe sets.
    
    Args:
        safe_sets: List of safe set polytopes
        
    Returns:
        List of unsafe set polytopes
    """
    # This is a placeholder implementation
    # In practice, this would convert the union of safe sets to an equivalent
    # representation as the union of unsafe sets
    # For now, just return the original sets as a placeholder
    return safe_sets


def aux_combineSafeSets(spec: List[Specification]) -> List[Specification]:
    """
    Combine all specifications involving safe sets to a single safe set.
    
    Args:
        spec: List of specifications
        
    Returns:
        Combined specifications
    """
    # find all specifications that define a safe set
    ind = []
    for i in range(len(spec)):
        if spec[i].type == 'safeSet':
            ind.append(i)
    
    # check if safe sets exist
    if len(ind) > 1:
        # combine safe sets to a single polytope
        poly = spec[ind[0]].set
        for i in range(1, len(ind)):
            poly = poly & spec[ind[i]].set
        
        specNew = Specification(poly, 'safeSet')
        
        # remove old specifications
        ind_ = [i for i in range(len(spec)) if i not in ind]
        
        if not ind_:
            spec = [specNew]
        else:
            spec = [spec[i] for i in ind_]
            spec = spec[0].add(specNew) if len(spec) == 1 else spec + [specNew]
    
    return spec
