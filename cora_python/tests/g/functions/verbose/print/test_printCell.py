import pytest
import numpy as np
from cora_python.g.functions.verbose.print.printCell import printCell
from cora_python.contSet.interval.interval import Interval

def test_printCell(capsys):
    # Test with an empty list
    printCell([])
    captured = capsys.readouterr()
    assert "{}" in captured.out

    # Test with a simple list
    printCell([1, "text", np.array([1, 2, 3])])
    captured = capsys.readouterr()
    assert "1" in captured.out
    assert "'text'" in captured.out
    assert "[1 2 3]" in captured.out

    # Test with a nested list (cell of cells)
    C = [1, "text"]
    nested_list = [C, C, C]
    printCell(nested_list)
    captured = capsys.readouterr()
    # Should print a 1x3 cell array
    assert "{[1] 'text'} {[1] 'text'} {[1] 'text'}" in captured.out or "{[1, 'text'], [1, 'text'], [1, 'text']}" in captured.out

    # Test with a 2D nested list
    nested_2d = [
        [1, "a"],
        [np.array([2]), Interval(3, 4)]
    ]
    printCell(nested_2d)
    captured = capsys.readouterr()
    assert "{[1] 'a'       }" in captured.out
    assert "{[2] interval }" in captured.out
    
    # Test different accuracies
    C_float = [1.12345, 2.67890]
    printCell(C_float, 'high')
    captured = capsys.readouterr()
    assert str(1.12345) in captured.out
    
    printCell(C_float, 'low')
    captured = capsys.readouterr()
    assert "1.1" in captured.out

    # Test compact printing
    C_large = [[i for i in range(20)] for _ in range(20)]
    printCell(C_large, 'high', do_compact=True)
    captured = capsys.readouterr()
    # In compact mode for a large cell array, it should print the size
    assert "20x20 cell" in captured.out

    printCell(C_large, 'high', do_compact=False)
    captured = capsys.readouterr()
    # Not compact, should print the full content
    assert "};" in captured.out 