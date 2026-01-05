from .nonlinearARX import NonlinearARX
from .display import display, display_

NonlinearARX.display = display
NonlinearARX.display_ = display_

# Attach display_ to __str__
NonlinearARX.__str__ = lambda self: display_(self)


__all__ = ['NonlinearARX'] 