from afunc import afunc

class A:

    def __init__(self):
        p = 50

    def __str__(self):
        return self.display()
    
    def func(self):
        return afunc()
    
    def display(self):
        return self.func()
    