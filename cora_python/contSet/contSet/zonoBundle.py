from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def zonoBundle(self, *varargin):
    """
    conversion to zonoBundle objects

    Syntax:
        zB = zonoBundle(S)

    Inputs:
        S - contSet object

    Outputs:
        zB - zonoBundle object
    """

    # is overridden in subclass if implemented; throw error
    raise CORAerror('CORA:noops', self) 