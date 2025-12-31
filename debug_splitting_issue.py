"""Debug the splitting issue with interval contractor"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing interval contractor with splits=2 ===")
print(f"Initial domain: {dom}")

# Simulate the exact splitting logic from contract.py
list_intervals = [dom]

for i in range(2):  # splits = 2
    print(f"\n--- Split iteration {i} ---")
    print(f"list_intervals at start: {list_intervals}")
    print(f"Number of intervals: {len(list_intervals)}")
    
    list_ = [None] * (2 * len(list_intervals))
    counter = 0
    
    # Loop over all sets in the list
    for j in range(len(list_intervals)):
        print(f"\n  Processing interval {j}: {list_intervals[j]}")
        
        # Determine the best dimension to split and split the domain
        domSplit = _aux_bestSplit(f, list_intervals[j])
        print(f"  Split result: {domSplit}")
        
        # Loop over all splitted domains
        for k in range(len(domSplit)):
            print(f"\n    Processing split {k}: {domSplit[k]}")
            
            # Check if the domain is empty
            temp = f(domSplit[k])
            print(f"    f(split[{k}]): {temp}")
            
            p = np.zeros(len(temp) if hasattr(temp, '__len__') else 1)
            print(f"    Checking if f(split[{k}]) contains {p}")
            
            if not temp.contains_(p):
                print(f"    SKIPPED: f(split[{k}]) does not contain 0")
                continue
            
            print(f"    Contracting split {k} with 'interval' algorithm...")
            # Contract the splitted domain (recursive call with splits=None)
            domTemp = contract(f, domSplit[k], 'interval', 2, None)
            print(f"    Contract result: {domTemp}")
            
            if domTemp is None:
                print(f"    SKIPPED: domTemp is None")
            elif domTemp.representsa_('emptySet', np.finfo(float).eps):
                print(f"    SKIPPED: domTemp is empty set")
            else:
                print(f"    ADDED to list at index {counter}")
                list_[counter] = domTemp
                counter += 1
    
    list_intervals = list_[:counter]
    print(f"\n  list_intervals after iteration {i}: {list_intervals}")
    print(f"  counter: {counter}")
    
    if counter == 0:
        print(f"  WARNING: No intervals survived iteration {i}!")

print(f"\n=== Final result ===")
if len(list_intervals) > 0:
    res = list_intervals[0]
    print(f"  Starting with: {res}")
    for i in range(1, len(list_intervals)):
        print(f"  Unioning with: {list_intervals[i]}")
        res = res | list_intervals[i]
        print(f"  Result: {res}")
    print(f"\nFinal result: {res}")
else:
    print("  No intervals to union, returning None")

