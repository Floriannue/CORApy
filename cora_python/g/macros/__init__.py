"""
CORA macros - Global configuration constants

This module contains global configuration constants that control
CORA behavior, similar to MATLAB global variables.
"""

from .CORAVERSION import CORAVERSION
from .CORA_WARNINGS_ENABLED import CORA_WARNINGS_ENABLED
from .DISPLAYDIM_MAX import DISPLAYDIM_MAX
from .CHECKS_ENABLED import CHECKS_ENABLED
from .CORAGITBRANCH import CORAGITBRANCH
from .CORAROOT import CORAROOT
from .VALIDATEOPTIONS_ERRORS import VALIDATEOPTIONS_ERRORS

__all__ = [
    'CORAVERSION',
    'CORA_WARNINGS_ENABLED',
    'DISPLAYDIM_MAX',
    'CHECKS_ENABLED',
    'CORAGITBRANCH',
    'CORAROOT',
    'VALIDATEOPTIONS_ERRORS'
]

# Global flag to enable/disable input argument checking
# Set to True by default for safety, can be disabled for performance
CHECKS_ENABLED = True 