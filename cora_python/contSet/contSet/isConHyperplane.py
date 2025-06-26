from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def isConHyperplane(S):
    """
    isConHyperplane - (DEPRECATED -> representsa)

    Syntax:
        res = isConHyperplane(S)

    Inputs:
        S - contSet object

    Outputs:
        res - true/false
    """

    CORAwarning('CORA:deprecated', 'function', 'contSet/isConHyperplane', 'CORA v2024',
                "When updating the code, please replace every function call 'isConHyperplane(S)' with 'representsa(S,'conHyperplane')'.",
                'This change was made in an effort to unify the syntax across all set representations.')
    
    return S.representsa('conHyperplane') 