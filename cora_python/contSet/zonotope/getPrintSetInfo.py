def getPrintSetInfo(self):
    """
    returns all information to properly print a set
    to the command window

    Syntax:
        [abbrev,propertyOrder] = getPrintSetInfo(S)

    Inputs:
        S - contSet

    Outputs:
        abbrev - set abbreviation
        propertyOrder - order of the properties
    """

    abbrev = 'Zonotope';
    propertyOrder = {'c', 'G'};

    return abbrev, propertyOrder 