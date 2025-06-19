from .polytope import Polytope

def copy(p: Polytope) -> Polytope:
    """
    Creates a copy of a Polytope object.

    This function utilizes the copy constructor of the Polytope class.

    Args:
        p: A Polytope object.

    Returns:
        A new Polytope object that is a copy of p.
    """
    return Polytope(p) 