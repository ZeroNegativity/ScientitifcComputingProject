import pytest
from modules import dqn_training, dqn_test_and_evaluate

@pytest.mark.parametrize(
    "l2, l3, learning_rate, gamma, epsilon, epochs, mem_size, batch_size, max_moves",
    [
        (200, 120, 0.001, 0.9, 0.3, 5000, 1000, 200, 50),
        (100, 60, 0.001, 0.9, 0.3, 5000, 1000, 200, 50),
        (100, 60, 0.001, 0.9, 0.3, 3000, 500, 100, 50),
        (100, 60, 0.001, 0.8, 0.5, 3000, 500, 100, 50),
        (100, 60, 0.01, 0.8, 0.5, 3000, 500, 100, 50)
    ],
    ids=[
        "Test 1: Default Parameters",
        "Test 2: Smaller Hidden Layer Sizes",
        "Test 3: Smaller Buffer size and Epoch",
        "Test 4: High Exploration Rate",
        "Test 5: Higher learning rate and Smaller Epoch"
    ]
)
def test_dqn_training_and_evaluation(l2, l3, learning_rate, gamma, epsilon, epochs, mem_size, batch_size, max_moves):
    model, _ = dqn_training(l2, l3, learning_rate, gamma, epsilon, epochs, mem_size, batch_size, max_moves)
    win_perc, average_moves = dqn_test_and_evaluate(model)
    print(f"win_perc: {win_perc}, average_moves: {average_moves}")  
    try:
        assert win_perc > 0.80
        assert average_moves < 5
        print(f"Test passed successfully for parameters: {locals()}")
    except AssertionError as e:
        print(f"Error: {e}")
        raise
    