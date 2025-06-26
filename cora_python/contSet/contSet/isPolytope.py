from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def isPolytope(S):
    """
    isPolytope - (DEPRECATED -> representsa)

    Syntax:
        res = isPolytope(S)

    Inputs:
        S - contSet object

    Outputs:
        res - true/false
    """

    CORAwarning('CORA:deprecated', 'function', 'contSet/isPolytope', 'CORA v2024',
                "When updating the code, please replace every function call 'isPolytope(S)' with 'representsa(S,'polytope')'.",
                'This change was made in an effort to unify the syntax across all set representations.')
    
    return S.representsa('polytope') 