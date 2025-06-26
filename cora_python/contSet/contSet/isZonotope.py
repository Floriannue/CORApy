from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def isZonotope(S):
    """
    isZonotope - (DEPRECATED -> representsa)

    Syntax:
        res = isZonotope(S)

    Inputs:
        S - contSet object

    Outputs:
        res - true/false
    """

    CORAwarning('CORA:deprecated', 'function', 'contSet/isZonotope', 'CORA v2024',
                "When updating the code, please replace every function call 'isZonotope(S)' with 'representsa(S,'zonotope')'.",
                'This change was made in an effort to unify the syntax across all set representations.')
    
    return S.representsa('zonotope') 