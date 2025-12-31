from .inputArgsLength import inputArgsLength
from .isequalFunctionHandle import isequalFunctionHandle

# Keep input_args_length as an alias for backward compatibility (deprecated)
from .inputArgsLength import inputArgsLength as input_args_length

__all__ = ['inputArgsLength', 'input_args_length', 'isequalFunctionHandle'] 