from ot_sparse_projection import misc


def assert_not_power_of_2(a):
    if misc.is_power_of_two(a):
        raise AssertionError("{} is a power of 2".format(a))


def assert_power_of_2(a):
    if not misc.is_power_of_two(a):
        raise AssertionError("{} is not a power of 2".format(a))


def test_is_power_of_two():
    assert_not_power_of_2(-1)
    assert_not_power_of_2(0)
    assert_not_power_of_2(3)
    assert_not_power_of_2(10)
    assert_not_power_of_2(6)
    assert_not_power_of_2(11)
    assert_not_power_of_2(1025)
    assert_not_power_of_2(2.1)
    assert_power_of_2(2)
    assert_power_of_2(2.)
    assert_power_of_2(1)
    assert_power_of_2(4)
    assert_power_of_2(64)
    assert_power_of_2(1024)
    assert_power_of_2(1 << 30)
