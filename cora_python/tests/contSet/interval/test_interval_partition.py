import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval

def test_interval_partition():
    # Example from MATLAB documentation
    I = Interval(np.array([[2], [3], [4]]), np.array([[5], [6], [7]]))
    splits = 2
    
    Isplit = I.partition(splits)
    
    # Expected number of partitions is splits^dim
    assert len(Isplit) == splits ** I.dim()
    
    # Check the bounds of a few partitions to verify correctness
    # First partition should be [2, 3.5], [3, 4.5], [4, 5.5]
    part_0 = Isplit[0]
    true_0 = Interval(np.array([[2.0], [3.0], [4.0]]), np.array([[3.5], [4.5], [5.5]]))
    assert part_0.isequal(true_0)
    
    # Last partition should be [3.5, 5], [4.5, 6], [5.5, 7]
    part_last = Isplit[-1]
    true_last = Interval(np.array([[3.5], [4.5], [5.5]]), np.array([[5.0], [6.0], [7.0]]))
    assert part_last.isequal(true_last)
    
    # Check total volume
    total_vol = sum(p.volume() for p in Isplit)
    assert np.isclose(total_vol, I.volume()) 