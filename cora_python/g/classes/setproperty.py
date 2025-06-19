# setproperty - object constructor for set properties
#
# Syntax:
#    prop = SetProperty()
#    prop = SetProperty(val)
#
# Inputs:
#    val - value for property
#
# Outputs:
#    obj - generated SetProperty object

class SetProperty:
    def __init__(self, val=None):
        self.val = val 