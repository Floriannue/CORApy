"""
priv_reportReachError - reports information to the user in case
   the reachable set explodes; in other cases, re-raise the error

Syntax:
    priv_reportReachError(ME, time, ind)

Inputs:
    ME - Exception object
    time - current time
    ind - current step
"""

def priv_reportReachError(ME, time, ind):
    if getattr(ME, 'args', None) and len(ME.args) > 0:
        msg = str(ME)
    else:
        msg = repr(ME)

    if hasattr(ME, 'identifier') and ME.identifier == 'CORA:reachSetExplosion':
        print("\n" + msg)
        print(f"  Step {ind} at time t={time}")
        print("The reachable sets until the current step are returned.\n")
    else:
        raise ME
