import numpy as np
import pytest
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
# Assuming radius is attached to Ellipsoid for now, due to __init__.py issues
# from cora_python.contSet.ellipsoid.radius import radius

class TestEllipsoidRandPoint:
    def test_randPoint_empty_set(self):
        """Test randPoint_ with an empty ellipsoid."""
        n = 2
        E_empty = Ellipsoid.empty(n)
        p = E_empty.randPoint_()
        assert p.size == 0 # Should be an empty array
        assert p.shape == (n, 0) # Correct shape for an empty result

    def test_randPoint_within_range_non_degenerate(self):
        """Test if random points for non-degenerate ellipsoid are within expected range."""
        E1 = Ellipsoid(np.array([[5.43878115, 12.49771836], [12.49771836, 29.66621173]]),
                       np.array([[-0.74450683], [3.58006475]]))
        n = E1.dim()
        N = 5 * n # Number of points as in MATLAB test
        assert self.aux_withinRange(E1, N)

    def test_randPoint_within_range_degenerate(self):
        """Test if random points for degenerate ellipsoid are within expected range."""
        Ed1 = Ellipsoid(np.array([[4.25333428, 0.63464002], [0.63464002, 0.09469464]]),
                        np.array([[-2.46536569], [0.27178687]]))
        n = Ed1.dim()
        N = 5 * n
        assert self.aux_withinRange(Ed1, N)

    def test_randPoint_within_range_zero_rank(self):
        """Test if random points for zero-rank ellipsoid (a point) are within expected range."""
        E0 = Ellipsoid(np.array([[0.0, 0.0], [0.0, 0.0]]),
                       np.array([[1.09869336], [-1.98843878]]))
        n = E0.dim()
        N = 5 * n
        # For a zero-rank ellipsoid (a point), randPoint_ should return the point itself N times.
        # The 'extreme' type might behave differently, but for now, the aux_withinRange should pass if it's just the point.
        assert self.aux_withinRange(E0, N)

    def aux_withinRange(self, E, N):
        """Helper function to check if all extreme points are between min(radius) and max(radius)."""
        # MATLAB: Y = randPoint(E,N,'extreme');
        # MATLAB: nY = sqrt(sum((Y-E.q).^2,1));
        # MATLAB: rE = radius(E,n);
        # MATLAB: IntR = interval(min(rE),max(rE));
        # MATLAB: res = all(contains(IntR,nY));

        # Python equivalent:
        Y = E.randPoint_(N, type='extreme')
        
        # Calculate Euclidean norm of points relative to ellipsoid center
        nY = np.linalg.norm(Y - E.q, axis=0) # Sum is 1, axis=0 is column sum
        
        # Get ellipsoid radii
        # E.radius() returns eigenvalues of Q^-1, sorted descending. So E.radius(n) would be all of them
        rE = E.radius(E.dim()) # Pass E.dim() to get all radii
        
        # min and max of radii define the range
        # In MATLAB, interval(min(rE),max(rE)) defines an interval [min_r, max_r]
        min_r = np.min(rE)
        max_r = np.max(rE)
        
        # Check if all nY values are within this range
        # For Python: Check if min_r <= nY <= max_r for all points
        # A small tolerance might be needed due to floating point arithmetic
        tolerance = 1e-9 # Same as used in other comparisons

        # The MATLAB contains(IntR, nY) checks if nY is within the interval
        # It's equivalent to min_r <= nY and nY <= max_r
        res = np.all((nY >= min_r - tolerance) & (nY <= max_r + tolerance))
        
        # Additional check from MATLAB for randPoint (if ellipsoid is a point, the points should be exactly E.q)
        if E.representsa_('point', np.finfo(float).eps):
            # For a point, the sampled points should be exactly the center
            res = res and np.allclose(Y, E.q) 

        return res 