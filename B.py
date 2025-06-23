from A import A
from bfunc import bfunc

class B(A):

    def __init__(self):
        super().__init__()

    def func(self):
        return bfunc()

b = B()
#print(B.p)
test = str(b)
print(test)

if test == "B":
    print("test passed. child class dont need own __str__ method")
else:
    print("test failed. child class needs own __str__ method")