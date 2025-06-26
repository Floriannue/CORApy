from cora_python.contDynamics.contDynamics import ContDynamics

def display(self):
    """
    Displays a nonlinearARX object on the command window
    """
    # Get display from parent
    parent_str = super(type(self), self).display()

    # Create own display
    res = [parent_str]
    res.append("Type: Nonlinear discrete-time ARX system")
    res.append(f"Sampling time: {self.dt}")
    res.append(f"Number of past time steps: {self.n_p}")
    
    # Represent the function handle
    fun_name = "anonymous function"
    if hasattr(self.mFile, '__name__'):
        fun_name = self.mFile.__name__
    res.append(f"Function handle: {fun_name}")

    return '\n'.join(res) 