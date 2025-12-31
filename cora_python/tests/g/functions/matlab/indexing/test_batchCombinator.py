"""
test_batchCombinator - unit test function for batchCombinator

Syntax:
    pytest cora_python/tests/g/functions/matlab/indexing/test_batchCombinator.py

Inputs:
    -

Outputs:
    res - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: batchCombinator

Authors:       Michael Eichelbeck
Written:       02-August-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import pytest
from cora_python.g.functions.matlab.indexing.batchCombinator import batchCombinator


class TestBatchCombinator:
    """Test class for batchCombinator functionality"""
    
    def test_batchCombinator_zero_combinations(self):
        """Test 1: 0 combinations"""
        # MATLAB: comb_state = struct;
        comb_state = {}
        # MATLAB: [batch, comb_state] = batchCombinator(10, int16(13), 3, comb_state);
        batch, comb_state = batchCombinator(10, 13, 3, comb_state)
        # MATLAB: assert(res & (isempty(batch) & comb_state.done==true));
        assert len(batch) == 0 or batch.size == 0
        assert comb_state.get('done', False) == True
    
    def test_batchCombinator_one_combination(self):
        """Test 2: 1 combination"""
        # MATLAB: comb_state = struct;
        comb_state = {}
        # MATLAB: [batch, comb_state] = batchCombinator(10, int16(10), 10, comb_state);
        batch, comb_state = batchCombinator(10, 10, 10, comb_state)
        # MATLAB: expected = [1 2 3 4 5 6 7 8 9 10];
        expected = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        # MATLAB: assert(res & (all(batch==expected) & comb_state.done==true));
        assert np.array_equal(batch, expected)
        assert comb_state.get('done', False) == True
    
    def test_batchCombinator_ten_combinations(self):
        """Test 3: 10 combinations"""
        # MATLAB: comb_state = struct;
        comb_state = {}
        
        # MATLAB: n = 5;
        n = 5
        # MATLAB: k = 3;
        k = 3
        
        # MATLAB: combs = zeros(10,3);
        combs = np.zeros((10, 3), dtype=int)
        # MATLAB: i = 1;
        i = 0  # Python 0-based indexing
        
        # MATLAB: while true
        while True:
            # MATLAB: [batch, comb_state] = batchCombinator(n, int16(k), 3, comb_state);
            batch, comb_state = batchCombinator(n, k, 3, comb_state)
            
            # MATLAB: % save results
            # MATLAB: for j=1:length(batch(:,1))
            for j in range(batch.shape[0]):
                # MATLAB: combs(i,:) = batch(j,:);
                combs[i, :] = batch[j, :]
                # MATLAB: i = i + 1;
                i += 1
            
            # MATLAB: % break once done
            # MATLAB: if comb_state.done == true
            if comb_state.get('done', False):
                # MATLAB: break;
                break
        
        # MATLAB: % Test if the correct number of combinations has been generated
        # MATLAB: % while their order is irrelevant
        
        # MATLAB: %are min and max indices correct
        # MATLAB: res_minmax = (max(max(combs)) == n);
        res_minmax = (np.max(combs) == n)
        # MATLAB: res_minmax = res_minmax & (min(min(combs)) == 1);
        res_minmax = res_minmax and (np.min(combs) == 1)
        
        # MATLAB: %are all rows unique?
        # MATLAB: unique_rows = unique(combs, "rows");
        unique_rows = np.unique(combs, axis=0)
        # MATLAB: res_unique = (size(unique_rows,1) == 10);
        res_unique = (unique_rows.shape[0] == 10)
        
        # MATLAB: %does each row have only combinations without repetitions?
        # MATLAB: for i=1:size(combs, 1)
        for idx in range(combs.shape[0]):
            # MATLAB: unique_entries = unique(combs(i,:));
            unique_entries = np.unique(combs[idx, :])
            # MATLAB: res_unique = res_unique & (size(unique_entries,2) == k);
            res_unique = res_unique and (len(unique_entries) == k)
        
        # MATLAB: assert(res & res_minmax & res_unique);
        assert res_minmax and res_unique


def test_batchCombinator():
    """Test function for batchCombinator method.
    
    Runs all test methods to verify correct implementation.
    """
    test = TestBatchCombinator()
    test.test_batchCombinator_zero_combinations()
    test.test_batchCombinator_one_combination()
    test.test_batchCombinator_ten_combinations()
    
    print("test_batchCombinator: all tests passed")
    return True


if __name__ == "__main__":
    test_batchCombinator()

