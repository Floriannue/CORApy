def zonoBundle(I):
    from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
    """
    Converts an interval into a zonotope bundle.
    """
    # An interval is converted to a single zonotope, which is then placed in a list
    # to create a zonoBundle.
    z = I.zonotope()
    return ZonoBundle([z]) 