from cora_python.contDynamics.contDynamics import ContDynamics

def display_(self):
    """
    Displays a nonlinearSysDT object (internal function that returns string)
    
    Returns:
        str: String representation
    """
    # Get display from parent
    from cora_python.contDynamics.contDynamics.display import display_ as parent_display
    parent_str = parent_display(self)

    # Create own display
    res = [parent_str]
    res.append("Type: Nonlinear discrete-time system")
    res.append(f"Sampling time: {self.dt}")
    
    # Represent the function handles
    dyn_fun_name = "anonymous"
    if hasattr(self.mFile, '__name__'):
        dyn_fun_name = self.mFile.__name__
    res.append(f"Dynamic function handle: {dyn_fun_name}")

    out_fun_name = "anonymous"
    if hasattr(self.out_mFile, '__name__'):
        out_fun_name = self.out_mFile.__name__
    res.append(f"Output function handle: {out_fun_name}")

    return '\n'.join(res)


def display(self):
    """
    Displays a nonlinearSysDT object on the command window (prints to stdout)
    """
    print(display_(self), end='') 