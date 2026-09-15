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
    sample_move_from_visits,
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

    def add_dirichlet_noise(self, alpha=1.0, epsilon=0.25):
        # mix Dirichlet noise into the children's priors: P = (1-eps)*p + eps*eta
        # call once on the (expanded) root during self-play for exploration
        actions = list(self.child_nodes)
        noise = torch.distributions.Dirichlet(
            torch.full((len(actions),), float(alpha))
        ).sample()
        for a, eta in zip(actions, noise):
            child = self.child_nodes[a]
            child.prior = (1 - epsilon) * child.prior + epsilon * eta.item()


class MCTS:
    predictor = None
    game = None
    root = None

    def __init__(
        self,
        net: AlphaZeroNet,
        game: Connect4,
        rollouts: int = 4,
        mode: str = "train",
    ):
        assert mode in ("train", "eval"), f"invalid mode: {mode}"
        self.predictor = net
        self.game = game
        self.rollouts = rollouts
        self.mode = mode
        self.exploration_cutoff = 12
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
        # reset game
        self.reset()
        # expand root & add dirichlet noise to facilitate exploration
        done = self.game.is_done()
        player = self.game.get_current_player()
        game = self.game.clone()
        self.select(self.root, game=game, done=done, player=player)
        self.root.add_dirichlet_noise()

        # init params
        final_outcome = 0.0
        current_buffer = []
        current_node = self.root
        depth = 1

        # ruthless during evals, exploratory during training
        if self.mode == "train":
            temperature = 1
        elif self.mode == "eval":
            temperature = 0

        while not done:
            player = self.game.get_current_player()
            done = self.game.is_done()
            for _ in range(self.rollouts):
                game = self.game.clone()
                self.select(current_node, game, done, player)

            state, encoded_state, mcts_policy_vec = self.get_buffer_elem(current_node)
            current_buffer.append((state, encoded_state, mcts_policy_vec))
            if depth > self.exploration_cutoff:
                temperature = 0

            # for optimal action, sample based on stats for
            # first 12 moves. Then act greedy
            valid_actions = self.game.get_valid_actions()
            child_visits = {
                a: current_node.child_nodes[a].get_visit_count() for a in valid_actions
            }
            optimal_action = sample_move_from_visits(child_visits, temperature)

            _, done, reward = self.game.step(optimal_action)
            current_node = current_node.child_nodes[optimal_action]
            depth += 1

            # if done then collect the final reward and update
            # the current_buffer
            if done:
                current_buffer.append(self.get_buffer_elem(current_node))
                final_outcome = reward
                current_buffer = self.update_buffer_with_final_outcome(
                    final_outcome, current_buffer
                )
                break

        if len(current_buffer) > 0:
            return current_buffer
        else:
            print("buffer is empty")
            return []

    def update_buffer_with_final_outcome(self, final_outcome, current_buffer):
        for idx in range(len(current_buffer) - 1, -1, -1):
            current_buffer[idx] = (
                current_buffer[idx][0],
                current_buffer[idx][1],
                current_buffer[idx][2],
                final_outcome,
            )
            final_outcome = -final_outcome
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

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


if __name__ == "__main__":
    game = Connect4()
    net = AlphaZeroNet()
    mcts = MCTS(net, game, 200)
    buf = mcts.simulation()

    path = next_log_path()
    with open(path, "w") as f:
        json.dump({"frames": buffer_to_frames(buf)}, f, indent=2)
    print(f"wrote {len(buf)} frames to {path}")
