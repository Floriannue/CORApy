def VALIDATEOPTIONS_ERRORS():
    """
    VALIDATEOPTIONS_ERRORS - macro deciding whether validateOptions throws
    errors if model parameters or algorithm parameters are wrongly
    defined; otherwise, a message is printed on the command window
    CAUTION: the setting 'false' is discouraged and should only ever be
             used during development

    Returns:
        bool: True if errors should be thrown, False otherwise.
    """
    return True 