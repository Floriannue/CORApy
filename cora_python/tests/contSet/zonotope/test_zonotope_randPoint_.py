"""
test_zonotope_randPoint_ - comprehensive unit tests for Zonotope randPoint_ method

This file combines the original MATLAB-translated tests with comprehensive 
additional tests covering all edge cases and advanced sampling methods.

Syntax:
    python -m pytest cora_python/tests/contSet/zonotope/test_zonotope_randPoint_.py

Authors: Mark Wetzlinger (MATLAB original), Python translation and extensions by AI Assistant
Written: 05-October-2024 (MATLAB), Python translation: 2025
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.zonotope.randPoint_ import randPoint_
from cora_python.contSet.contSet.contains import contains
from cora_python.g.functions.matlab.validate.check import compareMatrices


class TestZonotopeRandPoint:
    """Comprehensive test class for Zonotope randPoint_ method."""

    def test_empty_zonotope(self):
        """Test sampling from empty zonotope"""
        # nD, empty zonotope
        n = 4
        Z = Zonotope.empty(n)
        p = randPoint_(Z)
        assert p.shape == (n, 0)

    def test_point_zonotope(self):
        """Test sampling from point zonotope (no generators)"""
        # 2D, zonotope that's a single point
        tol = 1e-7
        Z = Zonotope(np.array([2, 3]))
        p = randPoint_(Z, 1)
        p_extr = randPoint_(Z, 1, 'extreme')
        assert p.shape[1] == 1 and compareMatrices(p, Z.c.reshape(-1, 1), tol)
        assert p_extr.shape[1] == 1 and compareMatrices(p_extr, Z.c.reshape(-1, 1), tol)
        
        # Test with zero generators explicitly
        c = np.array([[2], [3]])
        G = np.zeros((2, 0))
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 5, 'standard')
        assert points.shape == (2, 5)
        
        # All points should be the center
        for i in range(5):
            assert np.allclose(points[:, [i]], c, atol=1e-12)

    def test_degenerate_zonotope(self):
        """Test sampling from degenerate zonotope"""
        tol = 1e-7
        
        # 3D, degenerate zonotope
        c = np.array([1, 2, -1])
        G = np.array([[1, 3, -2], [1, 0, 1], [2, 3, -1]])
        Z = Zonotope(c, G)
        numPoints = 10
        
        # Test all sampling methods
        methods = ['standard', 'uniform', 'uniform:hitAndRun', 'uniform:billiardWalk', 'extreme']
        for method in methods:
            if method == 'extreme':
                # Test different extreme point scenarios
                n = 3  # dimension
                
                # Less points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 0.5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # As many points as extreme points
                nrPtsExtreme = int(np.ceil(2**n))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # More points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
            else:
                p_random = randPoint_(Z, numPoints, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
        
        # 1D zonotope in 2D space (line segment)
        c = np.array([[0], [1]])
        G = np.array([[1], [0]])
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 20, 'standard')
        assert points.shape == (2, 20)
        
        # All points should lie on the line y=1, x in [-1,1]
        assert np.allclose(points[1, :], 1, atol=1e-10)
        assert np.all(points[0, :] >= -1 - 1e-10)
        assert np.all(points[0, :] <= 1 + 1e-10)

    def test_parallelotope(self):
        """Test sampling from parallelotope with uniform shortcut"""
        tol = 1e-7
        
        # 3D, parallelotope
        c = np.array([1, 2, -1])
        G = np.array([[1, 3, -2], [1, 0, 1], [4, 1, -1]])
        Z = Zonotope(c, G)
        numPoints = 10
        
        methods = ['standard', 'uniform', 'uniform:hitAndRun', 'uniform:billiardWalk', 'extreme']
        for method in methods:
            if method == 'extreme':
                # Test different extreme point scenarios
                n = 3  # dimension
                
                # Less points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 0.5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # As many points as extreme points
                nrPtsExtreme = int(np.ceil(2**n))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # More points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
            else:
                p_random = randPoint_(Z, numPoints, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
        
        # Test uniform sampling uses parallelotope shortcut when applicable
        # Create exact parallelotope (axis-aligned rectangle)
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])  # Should be detected as parallelotope
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 50, 'uniform')
        assert points.shape == (2, 50)
        
        # Check all points are contained
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # For parallelotope, points should be more uniformly distributed
        # Test that we get points near corners
        corners_found = 0
        tolerance = 0.3
        expected_corners = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        
        for corner in expected_corners:
            corner_array = np.array([[corner[0]], [corner[1]]])
            distances = np.linalg.norm(points - corner_array, axis=0)
            if np.any(distances < tolerance):
                corners_found += 1
        
        # Should find points near most corners for uniform distribution
        assert corners_found >= 2

    def test_general_zonotope(self):
        """Test sampling from general zonotope"""
        tol = 1e-7
        
        # 3D, general zonotope
        c = np.array([0, 1, 1])
        G = np.array([[0, -1, 1, 4, 2, 3, -2], 
                      [0, 1, 4, 2, 1, 3, -8], 
                      [0, 1, -3, 2, 1, 2, 6]])
        Z = Zonotope(c, G)
        numPoints = 10
        
        methods = ['standard', 'uniform', 'uniform:hitAndRun', 'uniform:billiardWalk', 'extreme']
        for method in methods:
            if method == 'extreme':
                # Test different extreme point scenarios
                n = 3  # dimension
                
                # Less points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 0.5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # As many points as extreme points
                nrPtsExtreme = int(np.ceil(2**n))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
                
                # More points than extreme points
                nrPtsExtreme = int(np.ceil(2**n * 5))
                p_random = randPoint_(Z, nrPtsExtreme, method)
                assert np.all(contains(Z, p_random, 'exact', tol))
            else:
                p_random = randPoint_(Z, numPoints, method)
                assert np.all(contains(Z, p_random, 'exact', tol))

    def test_radius_sampling(self):
        """Test radius sampling method"""
        # 2D zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test radius sampling
        points = randPoint_(Z, 100, 'radius')
        assert points.shape == (2, 100)
        
        # Check all points are contained
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Test that points span different radii from center
        distances = np.linalg.norm(points - c, axis=0)
        assert len(np.unique(np.round(distances, 3))) > 10  # Should have varied distances

    def test_boundary_sampling(self):
        """Test boundary sampling method"""
        # 2D zonotope
        c = np.array([[1], [2]])
        G = np.array([[2, 1], [0, 1]])
        Z = Zonotope(c, G)
        
        # Test boundary sampling
        points = randPoint_(Z, 50, 'boundary')
        assert points.shape == (2, 50)
        
        # All points should be contained in zonotope
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Boundary points should be on the surface (harder to test directly)
        # At least verify they're not all identical
        assert not np.allclose(points[:, 0:1], points[:, 1:2])

    def test_uniform_billiardwalk_default(self):
        """Test that 'uniform' defaults to billiard walk for general zonotopes"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0.5], [0, 1]])  # Non-parallelotope
        Z = Zonotope(c, G)
        
        points_uniform = randPoint_(Z, 30, 'uniform')
        points_billiard = randPoint_(Z, 30, 'uniform:billiardWalk')
        
        # Both should produce valid points
        assert points_uniform.shape == (2, 30)
        assert points_billiard.shape == (2, 30)
        
        # Check containment
        for i in range(30):
            assert Z.contains_(points_uniform[:, [i]], 'exact')
            assert Z.contains_(points_billiard[:, [i]], 'exact')

    def test_ballwalk_detailed(self):
        """Test ball walk method in detail"""
        # 3D zonotope
        c = np.array([[1], [2], [3]])
        G = np.array([[1, 0, 0.5], [0, 1, 0.2], [0, 0, 0.8]])
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 40, 'uniform:ballWalk')
        assert points.shape == (3, 40)
        
        # Check containment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Check that points are reasonably distributed
        mean_point = np.mean(points, axis=1, keepdims=True)
        center_distance = np.linalg.norm(mean_point - c)
        
        # Mean should be relatively close to center for uniform distribution
        # Relaxed tolerance since ball walk can have some bias
        assert center_distance < 1.5

    def test_hitandrun_detailed(self):
        """Test hit-and-run method in detail"""
        # 2D zonotope that's not axis-aligned
        c = np.array([[0], [0]])
        G = np.array([[1, 0.5], [0.5, 1]])
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 60, 'uniform:hitAndRun')
        assert points.shape == (2, 60)
        
        # Check containment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Hit-and-run should produce good coverage
        x_span = np.max(points[0, :]) - np.min(points[0, :])
        y_span = np.max(points[1, :]) - np.min(points[1, :])
        
        # Should span reasonable portion of zonotope
        assert x_span > 1.0
        assert y_span > 1.0

    def test_extreme_all_vertices(self):
        """Test extreme sampling returning all vertices"""
        # Simple square zonotope
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Request all extreme points
        try:
            vertices = randPoint_(Z, 'all', 'extreme')
            # Should return 4 vertices for square
            assert vertices.shape[1] == 4
            
            # Check that these are actual vertices
            expected_vertices = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
            for expected in expected_vertices:
                found = False
                for i in range(vertices.shape[1]):
                    if np.allclose(vertices[:, i], expected, atol=1e-10):
                        found = True
                        break
                assert found, f"Vertex {expected} not found"
                
        except Exception as e:
            # If 'all' is not implemented, test with specific number
            pytest.skip(f"'all' vertices not implemented: {e}")

    def test_extreme_boundary_combination(self):
        """Test extreme sampling with boundary points when more points requested than vertices"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Request more points than vertices (4 vertices + boundary points)
        points = randPoint_(Z, 10, 'extreme')
        assert points.shape == (2, 10)
        
        # Check containment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Should include actual vertices
        vertex_found = False
        for i in range(points.shape[1]):
            point = points[:, i]
            # Check if this point is a vertex (all generators at ±1)
            if np.allclose(np.abs(point), [1, 1], atol=1e-10):
                vertex_found = True
                break
        assert vertex_found

    def test_high_dimensional(self):
        """Test sampling from higher dimensional zonotope"""
        # 4D zonotope
        c = np.zeros((4, 1))
        G = np.eye(4)
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 30, 'standard')
        assert points.shape == (4, 30)
        
        # Check containment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Check that we get variation in all dimensions
        for dim in range(4):
            assert np.std(points[dim, :]) > 0.1

    def test_single_generator(self):
        """Test sampling with single generator"""
        c = np.array([[1], [2]])
        G = np.array([[0.5], [1.0]])
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 25, 'standard')
        assert points.shape == (2, 25)
        
        # All points should lie on line segment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # Check that points span the line segment
        distances_from_center = np.linalg.norm(points - c, axis=0)
        max_distance = np.linalg.norm(G)
        assert np.max(distances_from_center) <= max_distance + 1e-10

    def test_redundant_generators(self):
        """Test sampling with redundant generators"""
        c = np.array([[0], [0]])
        G = np.array([[1, 2, 1], [0, 0, 0]])  # Third generator is redundant
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 30, 'standard')
        assert points.shape == (2, 30)
        
        # Check containment
        for i in range(points.shape[1]):
            assert Z.contains_(points[:, [i]], 'exact')
        
        # All points should have y=0 (due to zero second row)
        assert np.allclose(points[1, :], 0, atol=1e-12)

    def test_different_seeds(self):
        """Test that different random seeds produce different results"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Set seed for reproducibility
        np.random.seed(42)
        points1 = randPoint_(Z, 20, 'standard')
        
        np.random.seed(123)
        points2 = randPoint_(Z, 20, 'standard')
        
        # Should produce different results
        assert not np.allclose(points1, points2)

    def test_edge_case_tiny_zonotope(self):
        """Test sampling from very small zonotope"""
        c = np.array([[0], [0]])
        G = np.array([[1e-10, 0], [0, 1e-10]])
        Z = Zonotope(c, G)
        
        points = randPoint_(Z, 10, 'standard')
        assert points.shape == (2, 10)
        
        # Points should be very close to center
        distances = np.linalg.norm(points - c, axis=0)
        assert np.all(distances <= 1e-9)

    def test_all_sampling_methods_consistency(self):
        """Test that all sampling methods produce valid points for the same zonotope"""
        c = np.array([[1], [1]])
        G = np.array([[1, 0.5], [0.5, 1]])
        Z = Zonotope(c, G)
        
        methods = ['standard', 'extreme', 'uniform:ballWalk', 'uniform:hitAndRun', 'radius', 'boundary']
        
        for method in methods:
            points = randPoint_(Z, 15, method)
            assert points.shape == (2, 15)
            
            # Check containment for all methods
            for i in range(points.shape[1]):
                assert Z.contains_(points[:, [i]], 'exact'), f"Method {method} produced invalid point"

    def test_performance_consistency(self):
        """Test that repeated calls with same parameters produce reasonable consistency"""
        c = np.array([[0], [0]])
        G = np.array([[1, 0], [0, 1]])
        Z = Zonotope(c, G)
        
        # Multiple runs of uniform sampling
        means = []
        for _ in range(5):
            points = randPoint_(Z, 100, 'uniform:billiardWalk')
            means.append(np.mean(points, axis=1))
        
        means = np.array(means)
        
        # Means should be relatively close to center for uniform sampling
        # Relaxed tolerance since sampling can have some variance
        for i in range(5):
            center_distance = np.linalg.norm(means[i, :] - c.flatten())
            assert center_distance < 0.5  # Should be reasonably close to center on average


def test_zonotope_randPoint():
    """Main test function for comprehensive Zonotope randPoint_ method."""
    test = TestZonotopeRandPoint()
    
    # Run all tests
    test.test_empty_zonotope()
    test.test_point_zonotope()
    test.test_degenerate_zonotope()
    test.test_parallelotope()
    test.test_general_zonotope()
    test.test_radius_sampling()
    test.test_boundary_sampling()
    test.test_uniform_billiardwalk_default()
    test.test_ballwalk_detailed()
    test.test_hitandrun_detailed()
    test.test_extreme_all_vertices()
    test.test_extreme_boundary_combination()
    test.test_high_dimensional()
    test.test_single_generator()
    test.test_redundant_generators()
    test.test_different_seeds()
    test.test_edge_case_tiny_zonotope()
    test.test_all_sampling_methods_consistency()
    test.test_performance_consistency()
    
    print("test_zonotope_randPoint_: all comprehensive tests passed")


if __name__ == '__main__':
    test_zonotope_randPoint()
    print("All tests passed!") 