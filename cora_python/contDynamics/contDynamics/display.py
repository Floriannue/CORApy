def display(self):
    """
    Displays a contDynamics object on the command window.

    Returns:
        str: A string representation of the contDynamics object.
    """
    
    res = []
    res.append(f"Continuous dynamics: '{self.name}'")
    res.append(f"  number of dimensions: {self.nrOfDims}")
    res.append(f"  number of inputs: {self.nrOfInputs}")
    res.append(f"  number of outputs: {self.nrOfOutputs}")
    res.append(f"  number of disturbances: {self.nrOfDisturbances}")
    res.append(f"  number of noises: {self.nrOfNoises}")
    
    return '\n'.join(res) 