import numpy as np
from typing import Any, List, Union, Tuple

from .checkValueAttributes import checkValueAttributes
from cora_python.g.functions.matlab.validate.preprocessing import readNameValuePair
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
import logging

# Do not configure logging globally here; leave it to application/tests
# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Assume CHECKS_ENABLED is True for now, or needs to be imported from g.macros if it exists.
# For a more robust solution, CHECKS_ENABLED would be a configuration or global variable.
CHECKS_ENABLED = True

def inputArgsCheck(args: List[List[Any]]) -> None:
    """
    inputArgsCheck - checks input arguments of CORA functions for compliance
    with semantic requirements; if any violation is detected, an error is
    thrown, otherwise nothing happens

    Syntax:
       inputArgsCheck(inputArgs)

    Inputs:
       inputArgs - list of lists with structure
                     [ [input1,id,[classes],[attributes]];
                       [input2,id,[possibilities]];
                        ...; ];
                   to describe how input arguments should be like:
                      input1, input2, ... - value of input argument
                      id - identifier for attribute check ('att') or string
                           comparison ('str')
                      classes - (only id = 'att') admissible classes as a
                                list of strings
                      attributes - (only id = 'att') list of admissible attributes
                                 (check function checkValueAttributes for details)
                      possibilities - (only id = 'str') admissible strings as
                                      a list of strings

    Outputs:
       ---

    Example:
       # obj = Capsule(np.zeros((2,1)),np.ones((2,1)));
       # N = 5;
       # type = 'standard';
       # inputArgs = [ [obj, 'att',['cell','Capsule','other'],['nonempty']];
       #               [N,   'att',['numeric'],               ['positive']];
       #               [type,'str',['standard','gaussian']                ] ];
       # inputArgsCheck(inputArgs);

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: checkValueAttributes, readNameValuePair

    Authors:       Mingrui Wang, Mark Wetzlinger, Tobias Ladner
    Written:       30-May-2022
    Last update:   23-January-2024 (MW, exact match for strings)
                   03-March-2025 (TL, reworked using checkValueAttributes)
    Last revision: ---
    """

    def aux_countingNumber(i: int) -> str:
        # Simple helper for ordinal numbers, could be more elaborate
        suffixes = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
        if 10 <= i % 100 <= 20:
            return f"{i}th"
        else:
            return f"{i}{suffixes[i % 10]}"

    def aux_errMsgClassAttributes(class_names: Union[str, List[str]], attributes: Union[str, List[str]]) -> str:
        # Simplified for now, can be expanded to match MATLAB's detailed output
        if not isinstance(class_names, list):
            class_names = [class_names]
        if not isinstance(attributes, list):
            attributes = [attributes]

        text = "expected one of the following: "
        for c_idx, class_name in enumerate(class_names):
            text += f"{class_name}"
            if c_idx < len(attributes) and attributes[c_idx]:
                if isinstance(attributes[c_idx], list) and attributes[c_idx]:
                    text += f" ({', '.join(map(str, attributes[c_idx]))})"
                elif isinstance(attributes[c_idx], str) and attributes[c_idx]:
                    text += f" ({attributes[c_idx]})"
            if c_idx < len(class_names) - 1:
                text += ", "
        return text

    def aux_checkAtt(i: int, input_arg: List[Any], value: Any, log: logging.Logger) -> None:
        # check attribute
        # input_arg: [value,'att',classes,attributes]

        # read out classes
        classes = input_arg[2]
        if not isinstance(classes, list):
            classes = [classes]

        # read out attributes
        attributes = []
        if len(input_arg) >= 4:
            attributes = input_arg[3]
            if not isinstance(attributes, list):
                attributes = [attributes]

        # ensure cell of cell array (if MATLAB had nested cells for attributes)
        # For Python, if attributes can be a list of lists, handle it.
        # Simplified: if the first attribute entry is a list, assume nested structure.
        if attributes and isinstance(attributes[0], list):
            pass # Already in expected format
        else:
            attributes = [attributes] * len(classes) # Replicate for each class if not varying
        
        if len(classes) != len(attributes):
            # This part handles MATLAB's flexible input where attributes might be scalar
            # and applied to all classes, or vice-versa. Simplified for Python.
            if len(classes) == 1 and len(attributes) > 1:
                classes = classes * len(attributes)
            elif len(attributes) == 1 and len(classes) > 1:
                attributes = attributes * len(classes)
            else:
                raise CORAerror('CORA:specialError',f"Mismatch between given number of classes and attributes for {aux_countingNumber(i)} argument.")


        # check class and attributes
        resvec = [False] * len(classes)
        for j in range(len(classes)):
            current_class_name = classes[j]
            current_attributes = attributes[j]
            log.debug(f"aux_checkAtt: Checking value={value} against class={current_class_name}, attributes={current_attributes}")
            resvec[j] = checkValueAttributes(value, current_class_name, current_attributes)
            log.debug(f"aux_checkAtt: checkValueAttributes result for class {current_class_name}: {resvec[j]}")
    
        # gather results
        res = any(resvec)
        log.debug(f"aux_checkAtt: Overall resvec={resvec}, any(resvec)={res}")
        if not res:
            # find best guess for given class
            classresvec = []
            for cls_name in classes:
                if isinstance(cls_name, str):
                    try:
                        # Handle numpy.ndarray specifically
                        if cls_name == 'numpy.ndarray':
                            classresvec.append(isinstance(value, np.ndarray))
                        else:
                            classresvec.append(isinstance(value, eval(cls_name)))
                    except:
                        classresvec.append(False)
                else:
                    classresvec.append(False)
            idx = [k for k, x in enumerate(classresvec) if x]

            if len(idx) == 1:
                # show specific error message for guess
                text = aux_errMsgClassAttributes(classes[idx[0]], attributes[idx[0]])
            else:
                # show error message for all classes
                text = aux_errMsgClassAttributes(classes, attributes)

            # throw error
            raise CORAerror('CORA:wrongValue',aux_countingNumber(i), text)

    def aux_checkStr(i: int, input_arg: List[Any], value: Any, log: logging.Logger) -> None:
        # read string
        if isinstance(input_arg[2], list):
            validateStr = input_arg[2]
        else:
            validateStr = [input_arg[2]]

        # Ensure 'value' is a scalar string if it's meant to be a single option.
        # This handles cases where 'value' might incorrectly be passed as ['single_option']
        actual_value = value
        if isinstance(value, list) and len(value) == 1:
            actual_value = value[0]

        # check exact match with admissible values
        if actual_value not in validateStr:
            # generate string of admissible values (user info)
            validrange = f"'" + "', '".join(validateStr) + "'"

            # throw error
            raise CORAerror('CORA:wrongValue', aux_countingNumber(i),
                validrange)


    # check if disabled by macro
    if not CHECKS_ENABLED:
        return

    # number of input arguments to original function that have to be checked
    nrInputArgs = len(args)

    # loop over all input arguments
    for i in range(nrInputArgs):
        # read information about i-th input argument
        inputArg = args[i]
        # read value of i-th input argument
        value = inputArg[0]
        # read identifier ('att' or 'str')
        identifier = inputArg[1]

        # case distinction
        if identifier == 'att': # check classname (and attributes) in this case
            aux_checkAtt(i + 1, inputArg, value, logger) # i+1 to match MATLAB 1-indexing for error messages

        elif identifier == 'str': # check the strings in this case
            aux_checkStr(i + 1, inputArg, value, logger)

        else:
            raise CORAerror("CORA:wrongValue", 'second', "'att' or 'str'.") 