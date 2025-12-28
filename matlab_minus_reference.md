# MATLAB Minus Operator Reference for Zonotope

## MATLAB Implementation

### Base Implementation: `contSet/minus.m`

MATLAB does **NOT** have a specific `zonotope/minus.m` file. Instead, it uses the generic `contSet/minus.m`:

```matlab
function S = minus(S,p)
    if isnumeric(p)
        % subtrahend is numeric, call 'plus' with negated vector
        S = S + (-p);
    elseif isnumeric(S) && isa(p,'contSet')
        % minuend is a vector, subtrahend is a set
        S = -p + S;
    else
        % throw error for set - set
        throw(CORAerror('CORA:notSupported',...
            'Use minkDiff instead for set-set subtraction'));
    end
end
```

### How MATLAB Handles Different Cases:

1. **`Z - v`** (zonotope minus vector):
   - Calls `Z + (-v)` using `plus.m`
   - Implemented in `zonotope/plus.m`

2. **`v - Z`** (vector minus zonotope):
   - Calls `-Z + v` 
   - Uses `contSet/uminus.m` → calls `-1 * Z` → uses `mtimes.m`
   - Then uses `plus.m` to add the vector

3. **`Z1 - Z2`** (zonotope minus zonotope):
   - **NOT supported via `-` operator**
   - Must use `minkDiff(Z1, Z2)` instead
   - This is the Minkowski difference operation

### Key Points:

- **No `zonotope/minus.m` file exists** - uses `contSet/minus.m`
- **`Z - v`** → `Z + (-v)` via `plus.m`
- **`v - Z`** → `-Z + v` via `uminus.m` + `plus.m`
- **`Z1 - Z2`** → **NOT ALLOWED** - use `minkDiff` instead

### MATLAB Operator Overloading:

In MATLAB, the `-` operator calls:
- `minus(obj1, obj2)` for binary minus
- `uminus(obj)` for unary minus (negation)

The `contSet/minus.m` acts as a dispatcher that:
- Routes `S - p` to `S + (-p)`
- Routes `p - S` to `-S + p`
- Rejects `S1 - S2` with an error

