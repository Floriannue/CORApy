import numpy as np

def string(I: 'Interval') -> np.ndarray:
    """
    string - conversion of an interval to a string
    
    Syntax:
        str_array = string(I)
    
    Inputs:
        I - interval object
    
    Outputs:
        str_array - numpy array of strings
    
    Example:
        I = Interval(np.array([[-1],[-2]]), np.array([[2],[1]]))
        str_array = string(I)
    """
    shape = I.inf.shape
    str_array = np.empty(shape, dtype=object)

    # Use np.nditer for iterating over arbitrary dimensions
    it = np.nditer(I.inf, flags=['multi_index'], op_flags=['readonly'])
    while not it.finished:
        idx = it.multi_index
        str_array[idx] = f"[{I.inf[idx]},{I.sup[idx]}]"
        it.iternext()
            
    return str_array 