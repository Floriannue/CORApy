from .linearARX import LinearARX
from .display import display, display_

LinearARX.display = display
LinearARX.display_ = display_

# Attach display_ to __str__
LinearARX.__str__ = lambda self: display_(self)

__all__ = ['LinearARX'] 