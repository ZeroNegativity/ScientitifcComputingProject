import torch
import numpy as np
from collections import deque
from IPython.display import clear_output
from Gridworld import Gridworld
import random


def dqn_training(l2, l3, learning_rate, gamma, epsilon, epochs, mem_size, batch_size, max_moves):
    
    action_set= {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }
    l1 = 64
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    replay = deque(maxlen=mem_size)

    losses = []

    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0

        while status == 1:
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()

            if np.random.rand() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            action = action_set[action_]

            game.makeMove(action)
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            next_state = model(state2)
            next_state_ = next_state.data.numpy()

            reward = game.reward()

            if reward == -1 or reward == -10:
                Y = reward + (gamma * next_state_[0][np.argmax(next_state_)])
            else:
                Y = reward

            done = True if reward > 0 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if reward != -1 or mov > max_moves:
                status = 0
                mov = 0

    losses = np.array(losses)

    return model, losses


def dqn_test_model(model, mode='static', display=True):
    action_set= {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while status == 1:
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]

        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()

        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if i > 15:
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win, i  # Return both win status and number of moves

def dqn_test_and_evaluate(model, max_games=10000, mode='random', display=False):
    wins = 0
    total_moves = 0

    for i in range(max_games):
        win, moves = dqn_test_model(model, mode=mode, display=display)
        total_moves += moves
        if win:
            wins += 1

    win_perc = float(wins) / float(max_games)
    average_moves = float(total_moves) / float(max_games)

    return win_perc, average_moves