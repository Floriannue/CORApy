from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def isInterval(S):
    """
    isInterval - (DEPRECATED -> representsa)

    Syntax:
        res = isInterval(S)

    Inputs:
        S - contSet object

    Outputs:
        res - true/false
    """

    CORAwarning('CORA:deprecated', 'function', 'contSet/isInterval', 'CORA v2024',
                "When updating the code, please replace every function call 'isInterval(S)' with 'representsa(S,'interval')'.",
                'This change was made in an effort to unify the syntax across all set representations.')
    
    return S.representsa('interval') 