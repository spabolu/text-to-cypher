from cypher_slm.evaluation import compare_result_sets


def test_compare_result_sets_is_order_insensitive():
    expected = [{"name": "Alice", "count": 2}, {"name": "Bob", "count": 1}]
    actual = [{"name": "Bob", "count": 1}, {"name": "Alice", "count": 2}]
    assert compare_result_sets(expected, actual)
