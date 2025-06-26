from cora_python.contDynamics.contDynamics import ContDynamics

def display(self):
    """
    Displays a nonlinearSysDT object on the command window
    """
    # Get display from parent
    parent_str = super(type(self), self).display()

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