from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def isHyperplane(S):
    """
    isHyperplane - (DEPRECATED -> representsa)

    Syntax:
        res = isHyperplane(S)

    Inputs:
        S - contSet object

    Outputs:
        res - true/false
    """

    CORAwarning('CORA:deprecated', 'function', 'isHyperplane', 'CORA v2024',
                "When updating the code, please replace every function call 'isHyperplane(S)' with 'representsa(S,'hyperplane')'.",
                'This change was made in an effort to unify the syntax across all set representations.')
    
    return S.representsa('hyperplane') 