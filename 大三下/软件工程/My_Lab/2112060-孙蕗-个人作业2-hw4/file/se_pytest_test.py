import pytest
from io import StringIO
from unittest.mock import patch
from se_pytest_origin import detect_cycle, has_cycle, main

# 对 'detect_cycle' 函数的参数化测试用例
@pytest.mark.parametrize("circuit_devices, num_devices, expected", [
    ({1: ('AND', ['I1', 'I2']), 2: ('OR', ['I3', 'O1']), 3: ('NOT', ['O1'])}, 3, False),
    ({1: ('AND', ['I1', 'I2']), 2: ('OR', ['I3', 'O1']), 3: ('NOT', ['O2']), 4: ('NAND', ['O3', 'O4']), 5: ('OR', ['O1', 'O4'])}, 5, True),
    ({1: ('AND', ['I1', 'I2']), 2: ('OR', ['I3', 'I4']), 3: ('XOR', ['O1', 'I5']), 4: ('NAND', ['O2', 'I6']), 5: ('NOR', ['O3', 'I7'])}, 5, False),
    ({1: ('AND', ['I1', 'O5']), 2: ('OR', ['O1', 'I4']), 3: ('XOR', ['O2', 'I5']), 4: ('NAND', ['O3', 'I6']), 5: ('NOR', ['O4', 'I7'])}, 5, True)
])
def test_detect_cycle(circuit_devices, num_devices, expected):
    assert detect_cycle(circuit_devices, num_devices) == expected


# 对 'has_cycle' 函数的参数化测试用例
@pytest.mark.parametrize("graph, start_vertex, expected", [
    ({1: [2], 2: [3], 3: [1]}, 1, True),
    ({1: [2], 3: [4]}, 1, False),
    ({1: [2], 2: [3], 3: [1], 4: [5]}, 1, True)
])
def test_has_cycle(graph, start_vertex, expected):
    visited = [False] * (max(graph.keys()) + 1)
    stack = [False] * (max(graph.keys()) + 1)
    assert has_cycle(graph, start_vertex, visited, stack) == expected

# 对 'main' 函数的参数化测试用例
@pytest.mark.parametrize("inputs, expected_output", [
    ([
        '2', '2 3', 'AND 2 I1 O3', 'OR 2 O1 I2', 'XOR 2 O2 I2', '3 5',
        'XOR 2 I1 I2', 'XOR 2 O1 I3', 'AND 2 O1 I3', 'AND 2 I1 I2', 'OR 2 O3 O4',
        '4', '0 1 1', '1 0 1', '1 1 1', '0 0 0', '2 5 2', '2 5 2', '2 5 2', '2 5 2'
    ], ['LOOP', '1 0', '1 0', '1 1', '0 0']),
    ([
        '1', '3 3', 'AND 2 I1 I2', 'OR 2 I2 I3', 'XOR 2 I1 I3', '4',
        '1 0 1', '0 0 1', '1 1 1', '1 1 0', '2 1 3', '2 3 1', '3 3 2 1', '1 3'
    ], ['0 0', '1 0', '0 1 1', '1']),
    ([
        '1', '5 5',
        'XOR 2 I1 I2', 'NAND 2 O1 I3', 'AND 2 O1 I4', 'NOT 1 I5', 'OR 2 O3 O4',
        '3', '0 1 0 1 1', '0 0 1 1 1', '1 0 1 1 1', '2 5 2', '3 5 3 4', '4 5 4 1 3'
    ], ['1 1', '0 0 0', '1 0 1 1']),
])
@patch('builtins.input')
@patch('sys.stdout', new_callable=StringIO)
def test_main(mock_stdout, mock_input, inputs, expected_output):
    mock_input.side_effect = inputs
    main()
    output = mock_stdout.getvalue().strip().split('\n')
    assert output == expected_output

if __name__ == "__main__":
    pytest.main()