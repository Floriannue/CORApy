import numpy as np

class IntKaucher:
    """
    intKaucher - implementation of Kaucher arithmetic [1]. For demonstration
    purposes, we consider Example 1 in [2]

    Syntax:
        obj = intKaucher(ll,rl)

    Inputs:
        ll - left limit
        rl - right limit

    Outputs:
        obj - generated object

    Example:
        # function f
        f = lambda x: x**2 - x

        # compute gradient
        # In Python, you might use a library like SymPy for symbolic differentiation
        import sympy as sp
        x_sym = sp.Symbol('x')
        df_sym = sp.diff(f(x_sym), x_sym)
        df = sp.lambdify(x_sym, df_sym, 'numpy')

        # compute bounds for gradient
        from cora_python.contSet.interval.interval import interval
        I = interval(2,3)
        c = I.center()
        gr_interval = df(I)

        # compute inner-approximation of the range
        x = IntKaucher(3,2)
        gr = IntKaucher(gr_interval.inf, gr_interval.sup)

        res = f(c) + gr * (x - c)

    References:
       [1] E. Kaucher "Interval Analysis in the Extended Interval Space", 
           Springer 1980 
       [2] E. Goubault et al. "Forward Inner-Approximated Reachability of 
           Non-Linear Continuous Systems", HSCC 2017
    """

    def __init__(self, ll, rl):
        ll = np.array(ll)
        rl = np.array(rl)
        
        if ll.shape != rl.shape:
            # Assuming a CORAError class or similar will be implemented
            raise ValueError("Input arguments must have identical dimensions!")
            
        self.inf = ll
        self.sup = rl

    # Mathematical Operators ----------------------------------------------
    def __add__(self, other):
        if isinstance(other, IntKaucher):
            inf = self.inf + other.inf
            sup = self.sup + other.sup
            return IntKaucher(inf, sup)
        elif isinstance(other, (int, float, np.ndarray)):
            inf = self.inf + other
            sup = self.sup + other
            return IntKaucher(inf, sup)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return IntKaucher(-self.sup, -self.inf)

    def __sub__(self, other):
        return self.__add__(-other)

    def _aux_inP(self):
        return np.all(self.inf >= 0) and np.all(self.sup >= 0)

    def _aux_inMinusP(self):
        return np.all(self.inf <= 0) and np.all(self.sup <= 0)

    def _aux_inZ(self):
        return np.all(self.inf <= 0) and np.all(self.sup >= 0)

    def _aux_inDualZ(self):
        return np.all(self.inf >= self.sup)
        
    def __mul__(self, other):
        if isinstance(other, IntKaucher):
            # different cases of intervals
            if self._aux_inP() and other._aux_inDualZ():
                inf = self.inf * other.inf
                sup = self.inf * other.sup
            elif self._aux_inMinusP() and other._aux_inDualZ():
                inf = self.sup * other.sup
                sup = self.sup * other.inf
            elif self._aux_inZ() and other._aux_inDualZ():
                inf = 0
                sup = 0
            else:
                raise NotImplementedError("Desired multiplication not supported.")
            return IntKaucher(inf, sup)
        
        elif isinstance(other, (int, float, np.ndarray)):
            if np.all(other < 0):
                return (-self) * (-other)
            else:
                return IntKaucher(self.inf * other, self.sup * other)
        
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        # two scalars
        if self.isscalar() and other.isscalar():
           return self * other

        # scalar and matrix or matrix and scalar
        elif self.isscalar() or other.isscalar():
            if self.isscalar():
                scalar = self
                matrix = other
            else:
                scalar = other
                matrix = self
            
            res_inf = np.empty_like(matrix.inf)
            res_sup = np.empty_like(matrix.sup)
            
            for i in np.ndindex(matrix.inf.shape):
                res_val = scalar * IntKaucher(matrix.inf[i], matrix.sup[i])
                res_inf[i] = res_val.inf
                res_sup[i] = res_val.sup
            return IntKaucher(res_inf, res_sup)

        # matrix and matrix
        else:
            if self.inf.shape[1] != other.inf.shape[0]:
                raise ValueError("Matrix dimensions are not compatible for multiplication.")

            inf_res = np.zeros((self.inf.shape[0], other.inf.shape[1]))
            sup_res = np.zeros((self.inf.shape[0], other.inf.shape[1]))

            for i in range(self.inf.shape[0]):
                for j in range(other.inf.shape[1]):
                    temp = IntKaucher(0, 0)
                    for k in range(self.inf.shape[1]):
                        ik1 = IntKaucher(self.inf[i, k], self.sup[i, k])
                        ik2 = IntKaucher(other.inf[k, j], other.sup[k, j])
                        temp += ik1 * ik2
                    inf_res[i, j] = temp.inf
                    sup_res[i, j] = temp.sup
            
            return IntKaucher(inf_res, sup_res)

    # Additional Methods --------------------------------------------------
    def to_interval(self):
        from cora_python.contSet.interval.interval import interval
        return interval(self.inf, self.sup)

    def prop(self):
        """ makes a Kaucher interval proper (inf < sup) """
        return IntKaucher(np.minimum(self.inf, self.sup), np.maximum(self.inf, self.sup))

    def is_prop(self):
        """ checks if a Kaucher interval is proper (inf < sup) """
        return np.all(self.inf <= self.sup) # Note: MATLAB had >=, which seems incorrect for 'proper'

    def isscalar(self):
        return self.inf.ndim == 0 and self.sup.ndim == 0

    def shape(self):
        return self.inf.shape

    def __str__(self):
        if self.inf.ndim == 0:
            return f"[{self.inf:.5f}, {self.sup:.5f}]"
        
        rows, columns = self.inf.shape
        name = 'IntKaucher:'
        string = name + '\n'
        for i in range(rows):
            row_str = []
            for j in range(columns):
                row_str.append(f"[{self.inf[i, j]:.5f}, {self.sup[i, j]:.5f}]")
            string += " ".join(row_str) + "\n"
        return string

    def __repr__(self):
        return self.__str__() 