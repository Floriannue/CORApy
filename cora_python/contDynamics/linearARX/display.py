import numpy as np
from cora_python.contDynamics.contDynamics import ContDynamics
from cora_python.g.functions.verbose.display import display_matrix_vector

def display(self):
    """
    Displays a linearARX object on the command window
    """
    # Get display from parent
    parent_str = super(type(self), self).display()

    # Create own display
    res = [parent_str]
    res.append("Type: Linear discrete-time ARX system")
    res.append(f"Sampling time: {self.dt}")
    res.append("y(k) = sum_{i=1}^p A_bar{i} y(k-i) + sum_{i=1}^{p+1} B_bar{i} u(k-i+1)")
    
    res.append("Dimension:")
    res.append(display_matrix_vector(np.array(self.n_p), "p"))
    
    dim_display = min(self.n_p, 4)

    res.append("Output parameters:")
    for i in range(dim_display):
        res.append(display_matrix_vector(self.A_bar[i], f"A_bar{i+1}"))
        
    res.append("Input parameters:")
    for i in range(dim_display + 1):
        res.append(display_matrix_vector(self.B_bar[i], f"B_bar{i+1}"))

    return '\n'.join(res) 