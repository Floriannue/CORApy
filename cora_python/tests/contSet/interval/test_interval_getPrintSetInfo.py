import pytest
from cora_python.contSet import Interval


def test_get_print_set_info():
    i = Interval(0, 1)
    abbrev, prop_order = i.getPrintSetInfo()
    assert abbrev == 'I'
    assert prop_order == ['inf', 'sup'] 