"""
syntaxTree class 

Syntax:
    obj = syntaxTree(value,id)
    obj = syntaxTree(value,id,operator,funHan,nodes)

Inputs:
    value - interval or numeric value
    id - identifier/index for the variable
    operator - operator string ('+', '-', '*', 'power', etc.)
    funHan - function handle for backpropagation
    nodes - list of child nodes

Outputs:
    obj - generated syntaxTree object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval, polytope

Authors:       Matthias Althoff, Niklas Kochdumper
Written:       19-June-2015
Last update:   18-November-2015
                26-January-2016
                15-July-2017 (NK)
                04-November-2019 (ZL)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, List, Callable, Any, Union

from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.check.assertNarginConstructor import assertNarginConstructor
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues


class SyntaxTree:
    """
    Syntax tree class for representing mathematical expressions
    
    Properties:
        nodes: list of child nodes
        operator: operator string
        value: interval or numeric value
        funHan: function handle for backpropagation
        id: identifier/index for the variable
    """
    
    def __init__(self, value: Any, id_: Optional[int] = None, 
                 operator: Optional[str] = None, 
                 funHan: Optional[Callable] = None,
                 nodes: Optional[List] = None):
        """
        Class constructor
        
        Args:
            value: interval or numeric value
            id_: identifier/index for the variable
        """
        # Check number of input arguments
        assertNarginConstructor([2, 3, 4, 5], len([value, id_, operator, funHan, nodes]))
        
        # Parse input arguments
        defaults = [None, None, []]
        varargin = [operator, funHan, nodes]
        operator, funHan, nodes = setDefaultValues(defaults, varargin)
        
        # Assign values
        self.value = value
        self.id = id_
        self.operator = operator
        self.funHan = funHan
        self.nodes = nodes if nodes is not None else []
    
    def __pow__(self, exp: Union[int, float]) -> 'SyntaxTree':
        """
        Power operation for syntax tree
        MATLAB: res = syntaxTree(obj.value^exp,[],'power',fHan,{obj});
        """
        # MATLAB: For scalar syntaxTree, res = syntaxTree(obj.value^exp,[],'power',fHan,{obj});
        # Check if scalar (including Interval objects)
        is_scalar = (np.isscalar(self.value) or 
                    (hasattr(self.value, 'size') and self.value.size == 1) or
                    isinstance(self.value, Interval))
        
        if is_scalar:
            # Scalar case - MATLAB: res = syntaxTree(obj.value^exp,[],'power',fHan,{obj});
            fHan = lambda x, y: _aux_power_(x, exp, y)
            # MATLAB: obj.value^exp
            new_value = self.value ** exp if hasattr(self.value, '__pow__') else pow(self.value, exp)
            return SyntaxTree(new_value, None, 'power', fHan, [self])
        else:
            # Non-scalar case - MATLAB handles this with loops, but we simplify
            # For now, treat as scalar for each element
            fHan = lambda x, y: _aux_power_(x, exp, y)
            new_value = self.value ** exp if hasattr(self.value, '__pow__') else np.power(self.value, exp)
            return SyntaxTree(new_value, None, 'power', fHan, [self])
    
    def __rpow__(self, exp: Union[int, float]) -> 'SyntaxTree':
        """Reverse power operator"""
        fHan = lambda x, y: _aux_power_(x, exp, y)
        new_value = exp ** self.value if hasattr(exp, '__pow__') else pow(exp, self.value)
        return SyntaxTree(new_value, None, 'power', fHan, [self])
    
    def exp(self) -> 'SyntaxTree':
        """Exponential function"""
        fHan = lambda x, y: _aux_exp_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.exp(self.value.inf), np.exp(self.value.sup))
        else:
            new_value = np.exp(self.value)
        return SyntaxTree(new_value, None, 'exp', fHan, [self])
    
    def log(self) -> 'SyntaxTree':
        """Logarithm function"""
        fHan = lambda x, y: _aux_log_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.log(self.value.inf), np.log(self.value.sup))
        else:
            new_value = np.log(self.value)
        return SyntaxTree(new_value, None, 'log', fHan, [self])
    
    def sqrt(self) -> 'SyntaxTree':
        """Square root function"""
        fHan = lambda x, y: _aux_sqrt_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.sqrt(self.value.inf), np.sqrt(self.value.sup))
        else:
            new_value = np.sqrt(self.value)
        return SyntaxTree(new_value, None, 'sqrt', fHan, [self])
    
    def sin(self) -> 'SyntaxTree':
        """Sine function"""
        fHan = lambda x, y: _aux_sin_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.sin(self.value.inf), np.sin(self.value.sup))
        else:
            new_value = np.sin(self.value)
        return SyntaxTree(new_value, None, 'sin', fHan, [self])
    
    def cos(self) -> 'SyntaxTree':
        """Cosine function"""
        fHan = lambda x, y: _aux_cos_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.cos(self.value.inf), np.cos(self.value.sup))
        else:
            new_value = np.cos(self.value)
        return SyntaxTree(new_value, None, 'cos', fHan, [self])
    
    def tan(self) -> 'SyntaxTree':
        """Tangent function"""
        fHan = lambda x, y: _aux_tan_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.tan(self.value.inf), np.tan(self.value.sup))
        else:
            new_value = np.tan(self.value)
        return SyntaxTree(new_value, None, 'tan', fHan, [self])
    
    def asin(self) -> 'SyntaxTree':
        """Arcsine function"""
        fHan = lambda x, y: _aaux_sin_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arcsin(self.value.inf), np.arcsin(self.value.sup))
        else:
            new_value = np.arcsin(self.value)
        return SyntaxTree(new_value, None, 'asin', fHan, [self])
    
    def acos(self) -> 'SyntaxTree':
        """Arccosine function"""
        fHan = lambda x, y: _aaux_cos_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arccos(self.value.inf), np.arccos(self.value.sup))
        else:
            new_value = np.arccos(self.value)
        return SyntaxTree(new_value, None, 'acos', fHan, [self])
    
    def atan(self) -> 'SyntaxTree':
        """Arctangent function"""
        fHan = lambda x, y: _aaux_tan_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arctan(self.value.inf), np.arctan(self.value.sup))
        else:
            new_value = np.arctan(self.value)
        return SyntaxTree(new_value, None, 'atan', fHan, [self])
    
    def sinh(self) -> 'SyntaxTree':
        """Hyperbolic sine function"""
        fHan = lambda x, y: _aux_sinh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.sinh(self.value.inf), np.sinh(self.value.sup))
        else:
            new_value = np.sinh(self.value)
        return SyntaxTree(new_value, None, 'sinh', fHan, [self])
    
    def cosh(self) -> 'SyntaxTree':
        """Hyperbolic cosine function"""
        fHan = lambda x, y: _aux_cosh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.cosh(self.value.inf), np.cosh(self.value.sup))
        else:
            new_value = np.cosh(self.value)
        return SyntaxTree(new_value, None, 'cosh', fHan, [self])
    
    def tanh(self) -> 'SyntaxTree':
        """Hyperbolic tangent function"""
        fHan = lambda x, y: _aux_tanh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.tanh(self.value.inf), np.tanh(self.value.sup))
        else:
            new_value = np.tanh(self.value)
        return SyntaxTree(new_value, None, 'tanh', fHan, [self])
    
    def asinh(self) -> 'SyntaxTree':
        """Inverse hyperbolic sine function"""
        fHan = lambda x, y: _aaux_sinh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arcsinh(self.value.inf), np.arcsinh(self.value.sup))
        else:
            new_value = np.arcsinh(self.value)
        return SyntaxTree(new_value, None, 'asinh', fHan, [self])
    
    def acosh(self) -> 'SyntaxTree':
        """Inverse hyperbolic cosine function"""
        fHan = lambda x, y: _aaux_cosh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arccosh(self.value.inf), np.arccosh(self.value.sup))
        else:
            new_value = np.arccosh(self.value)
        return SyntaxTree(new_value, None, 'acosh', fHan, [self])
    
    def atanh(self) -> 'SyntaxTree':
        """Inverse hyperbolic tangent function"""
        fHan = lambda x, y: _aaux_tanh_(x, y)
        if isinstance(self.value, Interval):
            new_value = Interval(np.arctanh(self.value.inf), np.arctanh(self.value.sup))
        else:
            new_value = np.arctanh(self.value)
        return SyntaxTree(new_value, None, 'atanh', fHan, [self])
    
    def __add__(self, other: Any) -> 'SyntaxTree':
        """Addition operator"""
        if np.isscalar(self.value) and (np.isscalar(other) or 
            (isinstance(other, SyntaxTree) and np.isscalar(other.value))):
            # Both scalar
            fHan = lambda x, y, z: _aux_plus_(x, y, z)
            if isinstance(other, SyntaxTree):
                new_value = self.value + other.value
                return SyntaxTree(new_value, None, '+', fHan, [self, other])
            else:
                new_value = self.value + other
                return SyntaxTree(new_value, None, '+', fHan, [self, other])
        else:
            # Non-scalar case - handle element-wise
            if isinstance(other, SyntaxTree):
                # Check shapes if both are arrays/intervals
                if hasattr(self.value, 'shape') and hasattr(other.value, 'shape'):
                    if self.value.shape != other.value.shape:
                        raise CORAerror('CORA:dimensionMismatch', 'Dimensions must match')
                # Element-wise addition
                # MATLAB: res = syntaxTree(obj1.value + obj2.value,[],'+',fHan,{obj1,obj2});
                new_value = self.value + other.value
                return SyntaxTree(new_value, None, '+', lambda x, y, z: _aux_plus_(x, y, z), [self, other])
            else:
                # MATLAB: res = syntaxTree(obj1.value + obj2,[],'+',fHan,{obj1,obj2});
                # other is numeric, create syntaxTree node for it
                new_value = self.value + other
                other_node = SyntaxTree(other, None)
                return SyntaxTree(new_value, None, '+', lambda x, y, z: _aux_plus_(x, y, z), [self, other_node])
    
    def __radd__(self, other: Any) -> 'SyntaxTree':
        """Reverse addition operator"""
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> 'SyntaxTree':
        """
        Subtraction operator
        MATLAB: res = syntaxTree(obj1.value - obj2.value,[],'-',fHan,{obj1,obj2});
        """
        fHan = lambda x, y, z: _aux_minus_(x, y, z)
        # MATLAB: if isa(obj1,'syntaxTree') ... res = syntaxTree(obj1.value - obj2.value,[],'-',fHan,{obj1,obj2});
        if isinstance(other, SyntaxTree):
            # MATLAB: res = syntaxTree(obj1.value - obj2.value,[],'-',fHan,{obj1,obj2});
            new_value = self.value - other.value
            return SyntaxTree(new_value, None, '-', fHan, [self, other])
        else:
            # MATLAB: res = syntaxTree(obj1.value - obj2,[],'-',fHan,{obj1,obj2});
            # Use interval operator: self.value - other
            new_value = self.value - other
            other_node = SyntaxTree(other, None)
            return SyntaxTree(new_value, None, '-', fHan, [self, other_node])
    
    def __rsub__(self, other: Any) -> 'SyntaxTree':
        """
        Reverse subtraction operator
        MATLAB: res = syntaxTree(obj1 - obj2.value,[],'-',fHan,{obj1,obj2});
        """
        fHan = lambda x, y, z: _aux_minus_(x, y, z)
        # MATLAB: res = syntaxTree(obj1 - obj2.value,[],'-',fHan,{obj1,obj2});
        # Use interval operator: other - self.value will call Interval.__rsub__
        new_value = other - self.value
        # Store other as a syntaxTree with that value, or just store the value
        # MATLAB stores obj1 directly in nodes, so we need to create a syntaxTree for it
        if isinstance(other, SyntaxTree):
            other_node = other
        else:
            # Create a syntaxTree node for the numeric value
            other_node = SyntaxTree(other, None)
        return SyntaxTree(new_value, None, '-', fHan, [other_node, self])
    
    def __neg__(self) -> 'SyntaxTree':
        """Unary minus operator"""
        fHan = lambda x, y: _aux_uminus_(x, y)
        new_value = -self.value
        return SyntaxTree(new_value, None, 'uminus', fHan, [self])
    
    def __pos__(self) -> 'SyntaxTree':
        """Unary plus operator"""
        return self
    
    def __mul__(self, other: Any) -> 'SyntaxTree':
        """Multiplication operator"""
        if np.isscalar(self.value) and (np.isscalar(other) or 
            (isinstance(other, SyntaxTree) and np.isscalar(other.value))):
            # Both scalar
            fHan = lambda x, y, z: _aux_times_(x, y, z)
            if isinstance(other, SyntaxTree):
                new_value = self.value * other.value
                return SyntaxTree(new_value, None, '*', fHan, [self, other])
            else:
                new_value = self.value * other
                return SyntaxTree(new_value, None, '*', fHan, [self, other])
        else:
            # Non-scalar case
            if isinstance(other, SyntaxTree):
                if self.value.shape != other.value.shape:
                    raise CORAerror('CORA:dimensionMismatch', 'Dimensions must match')
                # Element-wise multiplication
                result = SyntaxTree.__new__(SyntaxTree)
                result.value = self.value * other.value
                result.id = self.id
                result.operator = '*'
                result.funHan = lambda x, y, z: _aux_times_(x, y, z)
                result.nodes = [self, other]
                return result
            else:
                # Scalar multiplication
                new_value = self.value * other
                return SyntaxTree(new_value, self.id, self.operator, self.funHan, self.nodes)
    
    def __rmul__(self, other: Any) -> 'SyntaxTree':
        """Reverse multiplication operator"""
        return self.__mul__(other)
    
    def backpropagation(self, value: 'Interval', int_: 'Interval') -> 'Interval':
        """
        Backpropagation method for interval contraction
        
        Args:
            value: target interval value
            int_: current interval domain
            
        Returns:
            res: contracted interval domain
        """
        # Check if node is base variable or not
        # MATLAB: if ~isempty(obj.id)
        if self.id is not None:
            # Base variable case
            # Check intersection
            # Access i-th dimension of interval
            int_id = Interval(int_.inf[self.id], int_.sup[self.id])
            
            # MATLAB: if ~isIntersecting(value,int(obj.id)) || representsa_(value,'emptySet',eps)
            if not int_id.isIntersecting_(value) or value.representsa_('emptySet', np.finfo(float).eps):
                raise CORAerror('CORA:emptySet')
            else:
                res = Interval(int_.inf.copy(), int_.sup.copy())
                res_val = value & int_id
                # Update the specific dimension
                if isinstance(res_val, Interval):
                    res.inf[self.id] = res_val.inf if np.isscalar(res_val.inf) else res_val.inf[0]
                    res.sup[self.id] = res_val.sup if np.isscalar(res_val.sup) else res_val.sup[0]
                else:
                    res.inf[self.id] = res_val
                    res.sup[self.id] = res_val
                return res
        else:
            # Non-base variable - apply operator
            # MATLAB: binary vs. unary operators
            # If no nodes and no id, this is a constant - return interval as-is
            if len(self.nodes) == 0:
                return int_
            elif len(self.nodes) == 1:
                # Unary operator
                # #region agent log
                import json
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"syntaxTree.py:backpropagation:unary_op","message":"Unary operator backpropagation","data":{"operator":self.operator,"value_inf":value.inf.tolist() if hasattr(value.inf, 'tolist') else str(value.inf),"value_sup":value.sup.tolist() if hasattr(value.sup, 'tolist') else str(value.sup),"int_inf":int_.inf.tolist(),"int_sup":int_.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                # Tighten interval
                val_ = self.funHan(value, self.nodes[0].value)
                # #region agent log
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    val_inf = val_.inf.tolist() if hasattr(val_, 'inf') and hasattr(val_.inf, 'tolist') else str(val_)
                    val_sup = val_.sup.tolist() if hasattr(val_, 'sup') and hasattr(val_.sup, 'tolist') else str(val_)
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"syntaxTree.py:backpropagation:after_unary_funHan","message":"After unary funHan call","data":{"val_inf":val_inf,"val_sup":val_sup},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                # Backpropagation for the node
                res = self.nodes[0].backpropagation(val_, int_)
                # #region agent log
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"syntaxTree.py:backpropagation:after_unary_recursive","message":"After unary recursive backpropagation","data":{"res_inf":res.inf.tolist(),"res_sup":res.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                return res
            else:
                # Binary operator
                # MATLAB: if isa(obj.nodes{1},'syntaxTree')
                if len(self.nodes) < 2:
                    # Not enough nodes for binary operation
                    return int_
                if isinstance(self.nodes[0], SyntaxTree):
                    # MATLAB: if isa(obj.nodes{2},'syntaxTree')
                    if isinstance(self.nodes[1], SyntaxTree):
                        # #region agent log
                        import json
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"syntaxTree.py:backpropagation:binary_op","message":"Binary operator backpropagation","data":{"operator":self.operator,"has_funHan":self.funHan is not None,"value_inf":value.inf.tolist() if hasattr(value.inf, 'tolist') else str(value.inf),"value_sup":value.sup.tolist() if hasattr(value.sup, 'tolist') else str(value.sup),"int_inf":int_.inf.tolist(),"int_sup":int_.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # MATLAB: [val1_,val2_] = obj.funHan(value,obj.nodes{1}.value,obj.nodes{2}.value);
                        # Both are syntax trees
                        val1_, val2_ = self.funHan(value, self.nodes[0].value, self.nodes[1].value)
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"syntaxTree.py:backpropagation:after_funHan","message":"After funHan call","data":{"val1_inf":val1_.inf.tolist() if hasattr(val1_, 'inf') and hasattr(val1_.inf, 'tolist') else str(val1_),"val1_sup":val1_.sup.tolist() if hasattr(val1_, 'sup') and hasattr(val1_.sup, 'tolist') else str(val1_),"val2_inf":val2_.inf.tolist() if hasattr(val2_, 'inf') and hasattr(val2_.inf, 'tolist') else str(val2_),"val2_sup":val2_.sup.tolist() if hasattr(val2_, 'sup') and hasattr(val2_.sup, 'tolist') else str(val2_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # MATLAB: res1 = backpropagation(obj.nodes{1},val1_,int);
                        # MATLAB: res2 = backpropagation(obj.nodes{2},val2_,int);
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            node1_op = getattr(self.nodes[0], 'operator', None) if hasattr(self.nodes[0], 'operator') else None
                            node2_op = getattr(self.nodes[1], 'operator', None) if hasattr(self.nodes[1], 'operator') else None
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"syntaxTree.py:backpropagation:before_recursive","message":"Before recursive backpropagation","data":{"node1_op":node1_op,"node2_op":node2_op,"val1_inf":val1_.inf.tolist() if hasattr(val1_, 'inf') and hasattr(val1_.inf, 'tolist') else str(val1_),"val1_sup":val1_.sup.tolist() if hasattr(val1_, 'sup') and hasattr(val1_.sup, 'tolist') else str(val1_),"val2_inf":val2_.inf.tolist() if hasattr(val2_, 'inf') and hasattr(val2_.inf, 'tolist') else str(val2_),"val2_sup":val2_.sup.tolist() if hasattr(val2_, 'sup') and hasattr(val2_.sup, 'tolist') else str(val2_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # Check if val1_ or val2_ represent impossible constraints
                        # (e.g., negative values for squares)
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            node1_has_id = hasattr(self.nodes[0], 'id') and self.nodes[0].id is not None
                            node2_has_id = hasattr(self.nodes[1], 'id') and self.nodes[1].id is not None
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"syntaxTree.py:backpropagation:before_node1","message":"Before backpropagating to node1","data":{"node1_op":node1_op,"node1_has_id":node1_has_id,"node1_id":self.nodes[0].id if node1_has_id else None,"val1_inf":val1_.inf.tolist() if hasattr(val1_, 'inf') and hasattr(val1_.inf, 'tolist') else str(val1_),"val1_sup":val1_.sup.tolist() if hasattr(val1_, 'sup') and hasattr(val1_.sup, 'tolist') else str(val1_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        try:
                            res1 = self.nodes[0].backpropagation(val1_, int_)
                        except CORAerror as e:
                            if e.identifier == 'CORA:emptySet' or e.identifier == 'CORA:outOfDomain':
                                # #region agent log
                                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"syntaxTree.py:backpropagation:node1_error","message":"Node1 raised error","data":{"error_id":e.identifier,"val1_inf":val1_.inf.tolist() if hasattr(val1_, 'inf') and hasattr(val1_.inf, 'tolist') else str(val1_),"val1_sup":val1_.sup.tolist() if hasattr(val1_, 'sup') and hasattr(val1_.sup, 'tolist') else str(val1_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                                raise e
                            raise
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"syntaxTree.py:backpropagation:before_node2","message":"Before backpropagating to node2","data":{"node2_op":node2_op,"node2_has_id":node2_has_id,"node2_id":self.nodes[1].id if node2_has_id else None,"val2_inf":val2_.inf.tolist() if hasattr(val2_, 'inf') and hasattr(val2_.inf, 'tolist') else str(val2_),"val2_sup":val2_.sup.tolist() if hasattr(val2_, 'sup') and hasattr(val2_.sup, 'tolist') else str(val2_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        try:
                            res2 = self.nodes[1].backpropagation(val2_, int_)
                        except CORAerror as e:
                            if e.identifier == 'CORA:emptySet' or e.identifier == 'CORA:outOfDomain':
                                # #region agent log
                                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"syntaxTree.py:backpropagation:node2_error","message":"Node2 raised error","data":{"error_id":e.identifier,"val2_inf":val2_.inf.tolist() if hasattr(val2_, 'inf') and hasattr(val2_.inf, 'tolist') else str(val2_),"val2_sup":val2_.sup.tolist() if hasattr(val2_, 'sup') and hasattr(val2_.sup, 'tolist') else str(val2_)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                                raise e
                            raise
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"syntaxTree.py:backpropagation:after_recursive","message":"After recursive backpropagation","data":{"res1_inf":res1.inf.tolist(),"res1_sup":res1.sup.tolist(),"res2_inf":res2.inf.tolist(),"res2_sup":res2.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # MATLAB: res = res1 & res2;
                        res = res1 & res2
                        # #region agent log
                        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"syntaxTree.py:backpropagation:after_intersection","message":"After intersection","data":{"res_inf":res.inf.tolist(),"res_sup":res.sup.tolist()},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        return res
                    else:
                        # MATLAB: val_ = obj.funHan(value,obj.nodes{1}.value,obj.nodes{2});
                        # Only first is syntax tree, second is numeric
                        val_ = self.funHan(value, self.nodes[0].value, self.nodes[1])
                        
                        # MATLAB: res = backpropagation(obj.nodes{1},val_,int);
                        res = self.nodes[0].backpropagation(val_, int_)
                        return res
                else:
                    # MATLAB: [~,val_] = obj.funHan(value,obj.nodes{1},obj.nodes{2}.value);
                    # Only second is syntax tree, first is numeric
                    _, val_ = self.funHan(value, self.nodes[0], self.nodes[1].value)
                    
                    # MATLAB: res = backpropagation(obj.nodes{2},val_,int);
                    res = self.nodes[1].backpropagation(val_, int_)
                    return res


# Factory function for creating syntax trees
def syntaxTree(value: Any, id_: Optional[int] = None, 
               operator: Optional[str] = None,
               funHan: Optional[Callable] = None,
               nodes: Optional[List] = None) -> SyntaxTree:
    """
    Factory function to create syntax tree objects
    
    Args:
        value: interval or numeric value
        id_: identifier/index for the variable
        operator: operator string
        funHan: function handle for backpropagation
        nodes: list of child nodes
        
    Returns:
        SyntaxTree object
    """
    return SyntaxTree(value, id_, operator, funHan, nodes)


# Auxiliary functions for backpropagation -------------------------------------

def _aux_sin_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for sin backpropagation"""
    # Intersection with valid domain
    int_val = int_val & Interval(-1, 1)
    
    # Apply inverse function
    resTemp1 = Interval(np.arcsin(int_val.inf), np.arcsin(int_val.sup))
    resTemp2 = Interval(-resTemp1.sup + np.pi, -resTemp1.inf + np.pi)
    
    # Compute updated upper bound
    ind1 = int(np.ceil((intPrev.sup - resTemp1.sup) / (2 * np.pi)))
    ind2 = int(np.ceil((intPrev.sup - resTemp2.sup) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.sup)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.sup)):
        supNew = intPrev.sup
    else:
        diff1 = intPrev.sup - (resTemp1.sup + (ind1 - 1) * 2 * np.pi)
        diff2 = intPrev.sup - (resTemp2.sup + (ind2 - 1) * 2 * np.pi)
        if diff1 < diff2:
            supNew = resTemp1.sup + (ind1 - 1) * 2 * np.pi
        else:
            supNew = resTemp2.sup + (ind2 - 1) * 2 * np.pi
    
    # Compute updated lower bound
    ind1 = int(np.floor((intPrev.inf - resTemp1.inf) / (2 * np.pi)))
    ind2 = int(np.floor((intPrev.inf - resTemp2.inf) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.inf)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.inf)):
        infiNew = intPrev.inf
    else:
        diff1 = resTemp1.inf + (ind1 + 1) * 2 * np.pi - intPrev.inf
        diff2 = resTemp2.inf + (ind2 + 1) * 2 * np.pi - intPrev.inf
        if diff1 < diff2:
            infiNew = resTemp1.inf + (ind1 + 1) * 2 * np.pi
        else:
            infiNew = resTemp2.inf + (ind2 + 1) * 2 * np.pi
    
    # Intersect with previous value
    res = intPrev & Interval(infiNew, supNew)
    return res


def _aux_cos_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for cos backpropagation"""
    # Intersection with valid domain
    int_val = int_val & Interval(-1, 1)
    
    # Apply inverse function
    resTemp1 = Interval(np.arccos(int_val.inf), np.arccos(int_val.sup))
    resTemp2 = Interval(-resTemp1.sup, -resTemp1.inf)
    
    # Compute updated upper bound
    ind1 = int(np.ceil((intPrev.sup - resTemp1.sup) / (2 * np.pi)))
    ind2 = int(np.ceil((intPrev.sup - resTemp2.sup) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.sup)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.sup)):
        supNew = intPrev.sup
    else:
        diff1 = intPrev.sup - (resTemp1.sup + (ind1 - 1) * 2 * np.pi)
        diff2 = intPrev.sup - (resTemp2.sup + (ind2 - 1) * 2 * np.pi)
        if diff1 < diff2:
            supNew = resTemp1.sup + (ind1 - 1) * 2 * np.pi
        else:
            supNew = resTemp2.sup + (ind2 - 1) * 2 * np.pi
    
    # Compute updated lower bound
    ind1 = int(np.floor((intPrev.inf - resTemp1.inf) / (2 * np.pi)))
    ind2 = int(np.floor((intPrev.inf - resTemp2.inf) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.inf)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.inf)):
        infiNew = intPrev.inf
    else:
        diff1 = resTemp1.inf + (ind1 + 1) * 2 * np.pi - intPrev.inf
        diff2 = resTemp2.inf + (ind2 + 1) * 2 * np.pi - intPrev.inf
        if diff1 < diff2:
            infiNew = resTemp1.inf + (ind1 + 1) * 2 * np.pi
        else:
            infiNew = resTemp2.inf + (ind2 + 1) * 2 * np.pi
    
    # Intersect with previous value
    res = intPrev & Interval(infiNew, supNew)
    return res


def _aux_power_(int_val: 'Interval', exp: Union[int, float], intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for power backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # #region agent log
    import json
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        int_val_inf = int_val.inf if np.isscalar(int_val.inf) else int_val.inf[0] if int_val.inf.size > 0 else 0
        int_val_sup = int_val.sup if np.isscalar(int_val.sup) else int_val.sup[0] if int_val.sup.size > 0 else 0
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"syntaxTree.py:_aux_power_:entry","message":"_aux_power_ called","data":{"exp":exp,"int_val_inf":int_val_inf,"int_val_sup":int_val_sup,"intPrev_inf":intPrev.inf.tolist() if hasattr(intPrev.inf, 'tolist') else str(intPrev.inf),"intPrev_sup":intPrev.sup.tolist() if hasattr(intPrev.sup, 'tolist') else str(intPrev.sup)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    if exp == 0:
        return intPrev
    
    # Check if interval is empty
    if int_val.representsa_('emptySet', np.finfo(float).eps):
        raise CORAerror('CORA:emptySet')
    
    # Even vs. odd exponent
    if exp % 2 == 0:
        # Intersection with valid domain
        # Check if inf is less than 0 (handle array case)
        int_val_inf = int_val.inf if np.isscalar(int_val.inf) else int_val.inf[0] if int_val.inf.size > 0 else 0
        int_val_sup = int_val.sup if np.isscalar(int_val.sup) else int_val.sup[0] if int_val.sup.size > 0 else 0
        # #region agent log
        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"syntaxTree.py:_aux_power_:even_check","message":"Even power check","data":{"int_val_inf":int_val_inf,"int_val_sup":int_val_sup,"int_val_sup_lt_0":bool(int_val_sup < 0)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        if int_val_inf < 0:
            if int_val_sup < 0:
                # All negative - impossible for even powers, raise emptySet
                # MATLAB raises CORA:outOfDomain, but contractForwardBackward only catches CORA:emptySet
                # So we raise CORA:emptySet to match expected behavior
                # #region agent log
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"syntaxTree.py:_aux_power_:raising_emptySet","message":"Raising CORA:emptySet for all negative interval","data":{"int_val_inf":int_val_inf,"int_val_sup":int_val_sup},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                raise CORAerror('CORA:emptySet')
            else:
                int_val = Interval(0, int_val.sup)
        
        # Apply inverse function
        temp1 = np.power(int_val.inf, 1.0 / exp)
        temp2 = np.power(int_val.sup, 1.0 / exp)
        
        # Select correct solution
        if intPrev.sup > 0 and intPrev.inf < 0:
            m = max(temp1, temp2)
            res = Interval(-m, m)
        else:
            if intPrev.sup > 0:
                res = Interval(min(temp1, temp2), max(temp1, temp2))
            else:
                res = Interval(-max(temp1, temp2), -min(temp1, temp2))
    else:
        # Odd exponent
        # Apply inverse function
        temp1 = np.power(int_val.inf, 1.0 / exp)
        temp2 = np.power(int_val.sup, 1.0 / exp)
        res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    # Check if result is empty (MATLAB raises error if intersection is empty)
    if res.representsa_('emptySet', np.finfo(float).eps):
        # #region agent log
        with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
            log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"syntaxTree.py:_aux_power_:empty_result","message":"Empty result after intersection in _aux_power_","data":{"int_val_inf":int_val_inf if 'int_val_inf' in locals() else None,"int_val_sup":int_val_sup if 'int_val_sup' in locals() else None,"exp":exp},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        raise CORAerror('CORA:emptySet')
    return res


def _aux_plus_(int_val: 'Interval', intPrev1: Any, intPrev2: Any) -> tuple:
    """
    Auxiliary function for plus backpropagation
    MATLAB: function [res1,res2] = aux_plus_(int,intPrev1,intPrev2)
    """
    # #region agent log
    import json
    with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
        intPrev1_info = {"type":type(intPrev1).__name__}
        intPrev2_info = {"type":type(intPrev2).__name__}
        if isinstance(intPrev1, Interval):
            intPrev1_info["inf"] = intPrev1.inf.tolist() if hasattr(intPrev1.inf, 'tolist') else str(intPrev1.inf)
            intPrev1_info["sup"] = intPrev1.sup.tolist() if hasattr(intPrev1.sup, 'tolist') else str(intPrev1.sup)
        if isinstance(intPrev2, Interval):
            intPrev2_info["inf"] = intPrev2.inf.tolist() if hasattr(intPrev2.inf, 'tolist') else str(intPrev2.inf)
            intPrev2_info["sup"] = intPrev2.sup.tolist() if hasattr(intPrev2.sup, 'tolist') else str(intPrev2.sup)
        log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"syntaxTree.py:_aux_plus_:entry","message":"_aux_plus_ called","data":{"int_val_inf":int_val.inf.tolist() if hasattr(int_val.inf, 'tolist') else str(int_val.inf),"int_val_sup":int_val.sup.tolist() if hasattr(int_val.sup, 'tolist') else str(int_val.sup),"intPrev1":intPrev1_info,"intPrev2":intPrev2_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    # MATLAB: if isa(intPrev1,'interval')
    if isinstance(intPrev1, Interval):
        # MATLAB: if isa(intPrev2,'interval')
        if isinstance(intPrev2, Interval):
            # MATLAB: res1 = intPrev1 & (int - intPrev2);
            # MATLAB: res2 = intPrev2 & (int - intPrev1);
            diff1 = int_val - intPrev2
            diff2 = int_val - intPrev1
            # #region agent log
            with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"syntaxTree.py:_aux_plus_:before_intersection","message":"Before intersection","data":{"diff1_inf":diff1.inf.tolist() if hasattr(diff1.inf, 'tolist') else str(diff1.inf),"diff1_sup":diff1.sup.tolist() if hasattr(diff1.sup, 'tolist') else str(diff1.sup),"diff2_inf":diff2.inf.tolist() if hasattr(diff2.inf, 'tolist') else str(diff2.inf),"diff2_sup":diff2.sup.tolist() if hasattr(diff2.sup, 'tolist') else str(diff2.sup)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            res1 = intPrev1 & diff1
            res2 = intPrev2 & diff2
            # #region agent log
            with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"syntaxTree.py:_aux_plus_:after_intersection","message":"After intersection","data":{"res1_inf":res1.inf.tolist() if hasattr(res1.inf, 'tolist') else str(res1.inf),"res1_sup":res1.sup.tolist() if hasattr(res1.sup, 'tolist') else str(res1.sup),"res1_empty":res1.representsa_('emptySet', np.finfo(float).eps) if hasattr(res1, 'representsa_') else False,"res2_inf":res2.inf.tolist() if hasattr(res2.inf, 'tolist') else str(res2.inf),"res2_sup":res2.sup.tolist() if hasattr(res2.sup, 'tolist') else str(res2.sup),"res2_empty":res2.representsa_('emptySet', np.finfo(float).eps) if hasattr(res2, 'representsa_') else False},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            # Check if either result is empty (MATLAB raises error if intersection is empty)
            if res1 is not None and hasattr(res1, 'representsa_') and res1.representsa_('emptySet', np.finfo(float).eps):
                # #region agent log
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"syntaxTree.py:_aux_plus_:res1_empty","message":"res1 is empty in _aux_plus_","data":{"int_val_inf":int_val.inf.tolist() if hasattr(int_val.inf, 'tolist') else str(int_val.inf),"int_val_sup":int_val.sup.tolist() if hasattr(int_val.sup, 'tolist') else str(int_val.sup)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                raise CORAerror('CORA:emptySet')
            if res2 is not None and hasattr(res2, 'representsa_') and res2.representsa_('emptySet', np.finfo(float).eps):
                # #region agent log
                with open(r'c:\Bachelorarbeit\Translate_Cora\.cursor\debug.log', 'a') as log_file:
                    log_file.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"syntaxTree.py:_aux_plus_:res2_empty","message":"res2 is empty in _aux_plus_","data":{"int_val_inf":int_val.inf.tolist() if hasattr(int_val.inf, 'tolist') else str(int_val.inf),"int_val_sup":int_val.sup.tolist() if hasattr(int_val.sup, 'tolist') else str(int_val.sup)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                raise CORAerror('CORA:emptySet')
        else:
            # MATLAB: res1 = intPrev1 & (int - intPrev2);
            # MATLAB: res2 = [];
            # intPrev2 is numeric
            res1 = intPrev1 & (int_val - intPrev2)
            res2 = None
    else:
        # MATLAB: res1 = [];
        # MATLAB: res2 = intPrev2 & (int - intPrev1);
        # intPrev1 is numeric, intPrev2 should be interval
        res1 = None
        res2 = intPrev2 & (int_val - intPrev1)
    
    return res1, res2


def _aux_minus_(int_val: 'Interval', intPrev1: Any, intPrev2: Any) -> tuple:
    """
    Auxiliary function for minus backpropagation
    MATLAB: function [res1,res2] = aux_minus_(int,intPrev1,intPrev2)
    """
    if isinstance(intPrev1, Interval):
        if isinstance(intPrev2, Interval):
            # MATLAB: res1 = intPrev1 & (int + intPrev2);
            # MATLAB: res2 = intPrev2 & (intPrev1 - int);
            res1 = intPrev1 & (int_val + intPrev2)
            res2 = intPrev2 & (intPrev1 - int_val)
        else:
            # MATLAB: res1 = intPrev1 & (int + intPrev2);
            # MATLAB: res2 = [];
            res1 = intPrev1 & (int_val + intPrev2)
            res2 = None
    else:
        # MATLAB: res1 = [];
        # MATLAB: res2 = intPrev2 & (intPrev1 - int);
        res1 = None
        res2 = intPrev2 & (intPrev1 - int_val)
    
    return res1, res2


def _aux_uminus_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for uminus backpropagation"""
    res = (-int_val) & intPrev
    return res


def _aux_times_(int_val: 'Interval', intPrev1: Any, intPrev2: Any) -> tuple:
    """Auxiliary function for times backpropagation"""
    if isinstance(intPrev1, Interval):
        if isinstance(intPrev2, Interval):
            if not intPrev2.contains_(0):
                res1 = intPrev1 & (int_val / intPrev2)
            else:
                res1 = intPrev1
            if not intPrev1.contains_(0):
                res2 = intPrev2 & (int_val / intPrev1)
            else:
                res2 = intPrev2
        else:
            res1 = intPrev1 & (int_val / intPrev2)
            res2 = None
    else:
        res1 = None
        if intPrev1 == 0:
            res2 = intPrev2
        else:
            res2 = intPrev2 & (int_val / intPrev1)
    
    return res1, res2


def _aux_tan_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for tan backpropagation"""
    # Intersection with valid domain
    int_val = int_val & Interval(-1, 1)
    
    # Apply inverse function
    resTemp1 = Interval(np.arctan(int_val.inf), np.arctan(int_val.sup))
    resTemp2 = Interval(resTemp1.inf + np.pi, resTemp1.sup + np.pi)
    
    # Compute updated upper bound
    ind1 = int(np.ceil((intPrev.sup - resTemp1.sup) / (2 * np.pi)))
    ind2 = int(np.ceil((intPrev.sup - resTemp2.sup) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.sup)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.sup)):
        supNew = intPrev.sup
    else:
        diff1 = intPrev.sup - (resTemp1.sup + (ind1 - 1) * 2 * np.pi)
        diff2 = intPrev.sup - (resTemp2.sup + (ind2 - 1) * 2 * np.pi)
        if diff1 < diff2:
            supNew = resTemp1.sup + (ind1 - 1) * 2 * np.pi
        else:
            supNew = resTemp2.sup + (ind2 - 1) * 2 * np.pi
    
    # Compute updated lower bound
    ind1 = int(np.floor((intPrev.inf - resTemp1.inf) / (2 * np.pi)))
    ind2 = int(np.floor((intPrev.inf - resTemp2.inf) / (2 * np.pi)))
    
    temp1_val = resTemp1 + ind1 * 2 * np.pi
    temp2_val = resTemp2 + ind2 * 2 * np.pi
    if (isinstance(temp1_val, Interval) and temp1_val.contains_(intPrev.inf)) or \
       (isinstance(temp2_val, Interval) and temp2_val.contains_(intPrev.inf)):
        infiNew = intPrev.inf
    else:
        diff1 = resTemp1.inf + (ind1 + 1) * 2 * np.pi - intPrev.inf
        diff2 = resTemp2.inf + (ind2 + 1) * 2 * np.pi - intPrev.inf
        if diff1 < diff2:
            infiNew = resTemp1.inf + (ind1 + 1) * 2 * np.pi
        else:
            infiNew = resTemp2.inf + (ind2 + 1) * 2 * np.pi
    
    # Intersect with previous value
    res = intPrev & Interval(infiNew, supNew)
    return res


def _aux_exp_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for exp backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < 0:
        if int_val.sup < 0:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain 0.')
        else:
            int_val = Interval(0, int_val.sup)
    
    # Apply inverse function
    temp1 = np.log(int_val.inf)
    temp2 = np.log(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aux_log_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for log backpropagation"""
    # Apply inverse function
    temp1 = np.exp(int_val.inf)
    temp2 = np.exp(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aux_sqrt_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for sqrt backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < 0:
        if int_val.sup < 0:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain 0.')
        else:
            int_val = Interval(0, int_val.sup)
    
    # Apply inverse function
    temp1 = np.power(int_val.inf, 2)
    temp2 = np.power(int_val.sup, 2)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_sin_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for asin backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < -np.pi / 2:
        if int_val.sup < -np.pi / 2:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain -pi/2.')
        else:
            int_val = Interval(-np.pi / 2, np.pi / 2)
    
    # Apply inverse function
    temp1 = np.sin(int_val.inf)
    temp2 = np.sin(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_cos_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for acos backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < 0:
        if int_val.sup < 0:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain 0.')
        else:
            int_val = Interval(0, int_val.sup)
    
    # Apply inverse function
    temp1 = np.cos(int_val.inf)
    temp2 = np.cos(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_tan_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for atan backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < -np.pi / 2:
        if int_val.sup < -np.pi / 2:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain -pi/2.')
        else:
            int_val = Interval(-np.pi / 2, np.pi / 2)
    
    # Apply inverse function
    temp1 = np.tan(int_val.inf)
    temp2 = np.tan(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aux_sinh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for sinh backpropagation"""
    # Intersection with valid domain
    int_val = int_val & Interval(-1, 1)
    
    # Apply inverse function
    temp1 = np.arcsinh(int_val.inf)
    temp2 = np.arcsinh(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aux_cosh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for cosh backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < 1:
        if int_val.sup < 1:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain 1.')
        else:
            int_val = Interval(1, int_val.sup)
    
    # Apply inverse function
    temp1 = np.arccosh(int_val.inf)
    temp2 = np.arccosh(int_val.sup)
    
    if intPrev.sup > 0 and intPrev.inf < 0:
        m = max(temp1, temp2)
        res = Interval(-m, m)
    else:
        if intPrev.sup > 0:
            res = Interval(min(temp1, temp2), max(temp1, temp2))
        else:
            res = Interval(-max(temp1, temp2), -min(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aux_tanh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for tanh backpropagation"""
    # Intersection with valid domain
    int_val = int_val & Interval(-1, 1)
    
    # Apply inverse function
    temp1 = np.arctanh(int_val.inf)
    temp2 = np.arctanh(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_sinh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for asinh backpropagation"""
    # Apply inverse function
    temp1 = np.sinh(int_val.inf)
    temp2 = np.sinh(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_cosh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for acosh backpropagation"""
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    
    # Intersection with valid domain
    if int_val.inf < 0:
        if int_val.sup < 0:
            raise CORAerror('CORA:outOfDomain', 'validDomain',
                           'Interval has to contain 0.')
        else:
            int_val = Interval(0, int_val.sup)
    
    # Apply inverse function
    temp1 = np.cosh(int_val.inf)
    temp2 = np.cosh(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res


def _aaux_tanh_(int_val: 'Interval', intPrev: 'Interval') -> 'Interval':
    """Auxiliary function for atanh backpropagation"""
    # Apply inverse function
    temp1 = np.tanh(int_val.inf)
    temp2 = np.tanh(int_val.sup)
    
    res = Interval(min(temp1, temp2), max(temp1, temp2))
    
    # Intersect with previous value
    res = intPrev & res
    return res

