import pytest
from cora_python.g.functions.verbose.print.printInterval import printInterval
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning

def test_printInterval(capsys):
    # Create an interval object
    I = Interval(-1, 1)

    # Test that the function raises a warning and calls printSet
    with pytest.warns(CORAwarning) as record:
        printInterval(I)

    # Check that a warning was issued
    assert len(record) == 1
    assert "deprecated" in str(record[0].message)
    assert "printInterval" in str(record[0].message)
    assert "printSet" in str(record[0].message)

    # Check that the output from printSet was produced
    captured = capsys.readouterr()
    assert "interval" in captured.out
    assert "inf: -1" in captured.out
    assert "sup: 1" in captured.out 