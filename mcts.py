import json

import torch

from env.connect4 import Connect4
from env.constants import ACTION_SPACE
from model import AlphaZeroNet
from util import (
    buffer_to_frames,
    get_mock_predictor_val,
    next_log_path,
    puct_eval,
    re_normalize,
)


class Node:
    predictor_state_value = 0.0

    def __init__(self, prior):
        self.cV_v = 0.0
        self.N_v = 0
        self.prior = prior
        self.child_nodes = {}

    def expand(self, state, policy, state_value, valid_actions):
        self.state = state
        self.predictor_state_value = state_value
        # zero out the probabilities of invalid actions and renormalize policy vector
        if len(valid_actions) != ACTION_SPACE:
            mask = torch.zeros_like(policy, dtype=torch.bool)
            mask[:, valid_actions] = True
            policy = re_normalize(policy, mask)
        for a in valid_actions:
            self.child_nodes[a] = Node(prior=policy[0][a].item())
        self.increment_cum_state_value(state_value=state_value)
        self.increment_visits()

    def update_terminal_val(self, state, reward):
        self.state = state
        self.predictor_state_value = reward
        self.increment_cum_state_value(state_value=reward)
        self.increment_visits()

    def increment_cum_state_value(self, state_value):
        self.cV_v += state_value

    def increment_visits(self):
        self.N_v += 1

    def get_state_value(self):
        if self.N_v == 0:
            return 0
        return self.cV_v / self.N_v

    def get_prior(self):
        return self.prior

    def get_visit_count(self):
        return self.N_v

    def is_expanded(self):
        if len(self.child_nodes) == 0:
            return False
        return True


class MCTS:
    predictor = None
    game = None
    root = None

    def __init__(self, net: AlphaZeroNet, game: Connect4, mcts_sims: int = 4):
        self.predictor = net
        self.game = game
        self.MCTS_SIMS = mcts_sims
        # root node initialized, not expanded
        self.root = Node(0.0)

    def reset(self):
        self.game.reset()
        # reinitialize root
        self.root = Node(0.0)

    def select(self, node: Node, game: Connect4, done: bool, player):
        # if we reach last node, update the
        # state_value and visit count with true reward
        if done == True:
            # if in this condition, the game ended and its loser's turn now. So its always a -1 return val.
            ret_val = 0.0
            if player == 0:
                ret_val = 0.0
            else:
                # Game over & -ve reward for loser
                ret_val = -1.0
            # game.get_encoded_board_state() returns the terminal state
            node.update_terminal_val(game.get_encoded_board_state(), ret_val)
            # caller is the opponent and they get a positive reward for playing a winning move
            return -ret_val

        if not node.is_expanded():
            # if node is not expanded, then expand the node and return the predictor state val.
            predictor_state_val = self.expand(
                node, game.get_encoded_board_state(), game.get_valid_actions()
            )
            # whoever is playing inferenced the model for state_val. we return the -ve of the val to the opponent
            return -predictor_state_val
        else:
            # if we have an expanded node, then make a decision with PUCT selection on valid actions
            valid_actions = game.get_valid_actions()
            optimal_action = max(
                valid_actions,
                key=lambda a: puct_eval(
                    -node.child_nodes[a].get_state_value(),
                    node.child_nodes[a].get_prior(),
                    node.child_nodes[a].get_visit_count(),
                    node.get_visit_count(),
                ),
            )
            # step only happens after making optimal selection
            _, done, player = game.step(optimal_action)
            reward = self.select(node.child_nodes[optimal_action], game, done, player)
            node.increment_cum_state_value(reward)
            node.increment_visits()
            return -reward

    # expand the terminal node;
    # - Inference predictor for policy vector and state value
    # - Create child nodes and hand them priors
    # - Lazy initialize node with state value
    def expand(self, node: Node, state, valid_actions):
        # policy_vector, state_value = self.predictor(state)
        policy_vector, state_value = get_mock_predictor_val()
        node.expand(state, policy_vector, state_value, valid_actions)
        return state_value

    def simulation(self):
        self.reset()
        current_buffer = []
        current_node = self.root
        final_reward = 0.0
        i = 1
        done = self.game.is_done()
        while not done:
            print("Depth ", i)
            i += 1
            player = self.game.get_current_player()
            done = self.game.is_done()
            for iter in range(self.MCTS_SIMS):
                game = self.game.clone()
                self.select(current_node, game, done, player)
                print("     Iter ", iter)

            state, encoded_state, mcts_policy_vec = self.get_buffer_elem(current_node)
            current_buffer.append((state, encoded_state, mcts_policy_vec))

            # use PUCT to find the next action and take a step,
            # if done then collect the final reward and update
            # the current_buffer
            valid_actions = self.game.get_valid_actions()
            optimal_action = max(
                valid_actions,
                key=lambda a: puct_eval(
                    -current_node.child_nodes[a].get_state_value(),
                    current_node.child_nodes[a].get_prior(),
                    current_node.child_nodes[a].get_visit_count(),
                    current_node.get_visit_count(),
                ),
            )

            _, done, reward = self.game.step(optimal_action)
            current_node = current_node.child_nodes[optimal_action]
            if done:
                current_buffer.append(self.get_buffer_elem(current_node))
                final_reward = reward
                current_buffer = self.update_buffer_with_final_reward(
                    final_reward, current_buffer
                )
                break
        if len(current_buffer) > 0:
            return current_buffer
        else:
            print("buffer is empty")
            return []

    def update_buffer_with_final_reward(self, final_reward, current_buffer):
        for idx in range(len(current_buffer) - 1, -1, -1):
            current_buffer[idx] = (
                current_buffer[idx][0],
                current_buffer[idx][1],
                current_buffer[idx][2],
                final_reward,
            )
            final_reward = -final_reward
        return current_buffer

    def get_buffer_elem(self, current_node):
        state = self.game.get_state()
        encoded_state = self.game.get_encoded_board_state()
        mcts_policy_vec = [0.0] * ACTION_SPACE
        # collect sample stats only if it isn't a terminal state (game.is_done == False)
        # and the state has been explored at least once, else return 0 vector - there
        # is nothing no policy here
        if not self.game.is_done() and current_node.get_visit_count() > 1:
            valid_actions = self.game.get_valid_actions()
            for a in valid_actions:
                mcts_policy_vec[a] = current_node.child_nodes[a].get_visit_count() / (
                    current_node.get_visit_count() - 1
                )

        return state, encoded_state, mcts_policy_vec


if __name__ == "__main__":
    game = Connect4()
    net = AlphaZeroNet()
    mcts = MCTS(net, game, 200)
    buf = mcts.simulation()

    path = next_log_path()
    with open(path, "w") as f:
        json.dump({"frames": buffer_to_frames(buf)}, f, indent=2)
    print(f"wrote {len(buf)} frames to {path}")
