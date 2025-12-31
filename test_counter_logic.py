"""Test counter logic difference"""
# MATLAB: counter = 1; then list_{counter} = item; counter = counter + 1; then list_(1:counter-1)
# Python: counter = 0; then list_[counter] = item; counter += 1; then list_[:counter]

# Simulate MATLAB (1-indexed)
print("=== MATLAB-style counter (1-indexed) ===")
list_matlab = [None] * 10
counter_matlab = 1
items_to_add = ["item_0", "item_1", "item_2"]
for item in items_to_add:
    if item is not None:  # Simulating the condition check
        list_matlab[counter_matlab] = item
        counter_matlab = counter_matlab + 1
print(f"counter_matlab after loop: {counter_matlab}")
print(f"list_matlab[1:counter_matlab]: {list_matlab[1:counter_matlab]}")
print(f"list_matlab[1:counter_matlab-1]: {list_matlab[1:counter_matlab-1]}")  # MATLAB uses this

# Simulate Python (0-indexed)
print("\n=== Python-style counter (0-indexed) ===")
list_python = [None] * 10
counter_python = 0
items_to_add = ["item_0", "item_1", "item_2"]
for item in items_to_add:
    if item is not None:  # Simulating the condition check
        list_python[counter_python] = item
        counter_python += 1
print(f"counter_python after loop: {counter_python}")
print(f"list_python[:counter_python]: {list_python[:counter_python]}")

# Check if they're equivalent
matlab_result = [x for x in list_matlab[1:counter_matlab-1] if x is not None]
python_result = [x for x in list_python[:counter_python] if x is not None]
print(f"\nMATLAB result (filtered): {matlab_result}")
print(f"Python result (filtered): {python_result}")
print(f"Equivalent? {matlab_result == python_result}")

