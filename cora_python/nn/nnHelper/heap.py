"""
heap - min heap

This is a Python translation of the MATLAB CORA implementation.

Authors: MATLAB: Tobias Ladner
         Python: AI Assistant
"""

import numpy as np
from typing import List, Any, Union


class Heap:
    """
    Min heap implementation for neural network layer operations.
    
    This class implements a min heap data structure that can store entries
    with keys for efficient minimum element retrieval.
    """
    
    def __init__(self, heap_entries: List[Any]):
        """
        Constructor for heap.
        
        Args:
            heap_entries: list storing keys or objects with key field
        """
        self.entries = []
        self.last_idx = 0
        
        n = len(heap_entries)
        
        for i in range(n):
            entry = heap_entries[i]
            if not hasattr(entry, 'key'):
                # Create entry struct equivalent
                entry_struct = type('Entry', (), {'key': entry, 'i': i})()
                entry = entry_struct
            
            if not self.entries:
                self.entries = [None] * n
            
            self.insert(entry)
    
    def insert(self, entry: Any):
        """
        Insert an entry into the heap.
        
        Args:
            entry: entry to insert
        """
        self.last_idx += 1
        self.entries[self.last_idx - 1] = entry
        self._sift_up(self.last_idx - 1)
    
    def pop(self) -> Any:
        """
        Remove the current min entry from the heap.
        
        Returns:
            minEntry: the minimum entry
        """
        min_entry = self.entries[0]
        max_entry = self.entries[self.last_idx - 1]
        self.last_idx -= 1
        self.entries[0] = max_entry
        self._sift_down(0)
        return min_entry
    
    def replace_min(self, new_min_entry: Any) -> Any:
        """
        Replace the current min entry with the given entry.
        
        Args:
            new_min_entry: new entry to replace minimum
            
        Returns:
            old_min_entry: the old minimum entry
        """
        old_min_entry = self.entries[0]
        self.entries[0] = new_min_entry
        self._sift_down(0)  # check invariant
        return old_min_entry
    
    def min(self) -> Any:
        """
        Return the current min entry from the heap.
        
        Returns:
            min_entry: the minimum entry
        """
        return self.entries[0]
    
    def isempty(self) -> bool:
        """
        Check if heap is empty.
        
        Returns:
            res: True if heap is empty, False otherwise
        """
        return self.last_idx == 0
    
    def _sift_down(self, p_idx: int):
        """
        Sift the entry at index p_idx down until heap invariant is ok.
        
        Args:
            p_idx: parent index
        """
        while True:
            p_entry = self.entries[p_idx]
            # get children
            c1_idx = 2 * p_idx + 0
            c2_idx = 2 * p_idx + 1
            
            if c1_idx >= self.last_idx:
                # no child entry
                return
            if c2_idx >= self.last_idx:
                # only one child
                c2_idx = c1_idx
            
            c1_entry = self.entries[c1_idx]
            c2_entry = self.entries[c2_idx]
            
            # check lexicographic order
            keys = [p_entry.key, c1_entry.key, c2_entry.key]
            if hasattr(keys[0], '__len__') and len(keys[0]) > 1:
                # Handle array keys
                keys_array = np.array(keys)
                idx = np.lexsort(keys_array.T)
            else:
                # Handle scalar keys
                idx = np.argsort(keys)
            
            if idx[0] == 0:
                # heap invariant ok
                return
            else:
                if idx[0] == 1:
                    c_idx = c1_idx
                    c_entry = c1_entry
                else:
                    c_idx = c2_idx
                    c_entry = c2_entry
                
                self.entries[p_idx] = c_entry
                self.entries[c_idx] = p_entry
                
                p_idx = c_idx
    
    def _sift_up(self, c_idx: int):
        """
        Sift the entry at index c_idx up until heap invariant is ok.
        
        Args:
            c_idx: child index
        """
        while True:
            if c_idx == 0:
                # reached min entry
                return
            
            c_entry = self.entries[c_idx]
            p_idx = (c_idx - 1) // 2
            p_entry = self.entries[p_idx]
            
            # check lexicographic order
            keys = [p_entry.key, c_entry.key]
            if hasattr(keys[0], '__len__') and len(keys[0]) > 1:
                # Handle array keys
                keys_array = np.array(keys)
                idx = np.lexsort(keys_array.T)
            else:
                # Handle scalar keys
                idx = np.argsort(keys)
            
            if idx[0] == 0:
                # heap invariant ok
                return
            else:
                self.entries[p_idx] = c_entry
                self.entries[c_idx] = p_entry
                
                c_idx = p_idx
