def CHECKS_ENABLED():
    """
    CHECKS_ENABLED - macro to enable/disable input argument checks (these
    checks are useful to ensure correct functionality and error tracing,
    but are time-consuming and therefore not required if the calling code
    does not contain any errors)

    Returns:
        bool: True if checks are enabled, False otherwise.
    """
    return True 