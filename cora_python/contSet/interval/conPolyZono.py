
def conPolyZono(I):
    """
    Convert an interval to a constrained polynomial zonotope.
    """
    from cora_python.contSet.conPolyZono.conPolyZono import ConPolyZono
    # First, convert the interval to a PolyZonotope using the interval's polyZonotope method
    pz = I.polyZonotope()
    
    # Then, convert the PolyZonotope to a ConPolyZono
    cpz = ConPolyZono(pz)
    
    return cpz 