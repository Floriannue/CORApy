from .nonlinearSysDT import NonlinearSysDT
from .display import display, display_

NonlinearSysDT.display = display
NonlinearSysDT.display_ = display_

# Attach display_ to __str__
NonlinearSysDT.__str__ = lambda self: display_(self)


__all__ = ['NonlinearSysDT'] 