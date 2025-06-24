from typing import Any

def reorder_numeric(S1: Any, S2: Any) -> tuple[Any, Any]:
    """
    Reorders the input arguments so that the first one is a class object
    and the second one is numeric.

    This occurs, e.g., if a contSet function is called with a numeric type
    as a first input argument and the respective contSet class as a second
    input argument.

    Args:
        S1: Class object or numeric.
        S2: Class object or numeric.

    Returns:
        A tuple where the first element is the class object and the second
        is the numeric value.
    """
    # Classic temporary-swap
    if isinstance(S1, (int, float, complex)):
        S1, S2 = S2, S1
    
    return S1, S2 