import pytest
import numpy as np
from cora_python.g.functions.verbose.print.printMatrix import printMatrix

def test_printMatrix(capsys):
    # Test with an empty matrix
    printMatrix(np.array([]))
    captured = capsys.readouterr()
    assert "[]" in captured.out

    # Test with scalars
    printMatrix(1)
    captured = capsys.readouterr()
    assert "1" in captured.out

    printMatrix(np.inf)
    captured = capsys.readouterr()
    assert "inf" in captured.out

    # Test with a row vector
    printMatrix(np.array([1, 2, 3]))
    captured = capsys.readouterr()
    assert "[1 2 3]" in captured.out

    # Test with a column vector
    printMatrix(np.array([[1], [2], [3]]))
    captured = capsys.readouterr()
    assert "[1; 2; 3]" in captured.out

    # Test with a 2D matrix
    M = np.array([[1, 2, 3], [4, 5, 6]])
    printMatrix(M)
    captured = capsys.readouterr()
    assert "[1 2 3; 4 5 6]" in captured.out

    # Test different accuracies
    M_float = np.array([[1.12345, 2.67890], [3.0, 4.5]])
    
    printMatrix(M_float, '%.4e')
    captured = capsys.readouterr()
    assert "1.1235e+00" in captured.out
    
    printMatrix(M_float, 'high')
    captured = capsys.readouterr()
    assert str(1.12345) in captured.out
    
    printMatrix(M_float, 'low')
    captured = capsys.readouterr()
    assert "1.1" in captured.out

    # Test compact printing
    M_large = np.random.rand(20, 20)
    printMatrix(M_large, 'high', do_compact=True)
    captured = capsys.readouterr()
    # In compact mode for a large matrix, it should print the size
    assert "rand(20,20)" in captured.out or "zeros(20,20)" in captured.out or "ones(20,20)" in captured.out

    printMatrix(M_large, 'high', do_compact=False)
    captured = capsys.readouterr()
    # Not compact, should print the full matrix content
    assert "];" in captured.out # check for multi-line output

    # Test sparse matrix
    from scipy.sparse import lil_matrix
    S = lil_matrix((10, 10))
    S[0, 1] = 4
    S[2, 3] = 5
    S[9, 9] = 9
    printMatrix(S)
    captured = capsys.readouterr()
    assert "sparse([1; 3; 10], [2; 4; 10], [4; 5; 9], 10, 10)" in captured.out

    # Test with braces
    printMatrix(M, use_braces=True)
    captured = capsys.readouterr()
    assert "{" in captured.out
    assert "}" in captured.out

    printMatrix(M, use_braces=False)
    captured = capsys.readouterr()
    assert "{" not in captured.out
    assert "}" not in captured.out 