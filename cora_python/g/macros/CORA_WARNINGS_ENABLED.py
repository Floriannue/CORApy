def CORA_WARNINGS_ENABLED(identifier='CORA:all'):
    """
    CORA_WARNINGS_ENABLED - specifies if a CORA warning should be shown

    Args:
        identifier (str): The identifier for the warning.

    Returns:
        bool: True if the warning is enabled, False otherwise.
    """
    
    warnings_config = {
        'CORA:all': True,
        # general folder warnings
        'CORA:app': True,
        'CORA:contDynamics': True,
        'CORA:contSet': True,
        'CORA:converter': True,
        'CORA:discDynamics': True,
        'CORA:examples': True,
        'CORA:global': True,
        'CORA:hybridDynamics': True,
        'CORA:manual': True,
        'CORA:matrixSets': True,
        'CORA:models': True,
        'CORA:nn': True,
        'CORA:specification': True,
        'CORA:unitTests': True,
        # special warnings
        'CORA:solver': True,
        'CORA:plot': True,
        'CORA:deprecated': True,
        'CORA:redundant': True,
        'CORA:interface': True,
    }

    if identifier not in warnings_config:
        # In MATLAB, this would throw a CORAerror.
        # The Python equivalent for error handling should be used here.
        # For now, we can raise a ValueError.
        raise ValueError(f"Unknown identifier '{identifier}'")

    return warnings_config[identifier] 