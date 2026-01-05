def display_(self):
    """
    Displays a contDynamics object (internal function that returns string).

    Returns:
        str: A string representation of the contDynamics object.
    """
    
    res = []
    res.append(f"Continuous dynamics: '{self.name}'")
    res.append(f"  number of dimensions: {self.nr_of_dims}")
    res.append(f"  number of inputs: {self.nr_of_inputs}")
    res.append(f"  number of outputs: {self.nr_of_outputs}")
    res.append(f"  number of disturbances: {self.nr_of_disturbances}")
    res.append(f"  number of noises: {self.nr_of_noises}")
    
    return '\n'.join(res)


def display(self):
    """
    Displays a contDynamics object on the command window (prints to stdout).
    """
    print(display_(self), end='') 