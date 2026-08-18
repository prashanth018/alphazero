from torch import Tensor

from env.connect4 import Connect4
from env.constants import ACTION_SPACE
from model import AlphaZeroNet
from util import puct_eval


class Node:
    cV_v = 0.0
    N_v = 0
    parent_state = 0
    parent_action = 0
    state = 0
    predictor_state_value = 0.0
    predictor_policy = []
    prior = 0.0
    child_nodes = {}

    def __init__(self, parent_state, parent_action, prior):
        self.cV_v = 0.0
        self.N_v = 0
        self.parent_state = parent_state
        self.parent_action = parent_action
        self.prior = prior

    def expand(self, state, policy, state_value):
        self.state = state
        self.predictor_policy = policy
        self.predictor_state_value = state_value
        for a in range(ACTION_SPACE):
            self.child_nodes[a] = Node(self.state, a, prior=policy[0][a].item())
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
            return True
        return False


class MCTS:
    predictor = None
    game = None
    root = None
    MCTS_SIMS = 1

    def __init__(self, net: AlphaZeroNet, game: Connect4):
        self.predictor = net
        self.game = game
        root_state = game.get_encoded_board_state()
        # root node initialized, not expanded
        self.root = Node(root_state, None, 0.0)

    def select(self, node: Node, game: Connect4, done: bool, player):
        # if we reach last node, update the
        # state_value and visit count with true reward
        if done == True:
            # if in this condition, the game ended and its loser's turn now. So its always a -1 return val.
            ret_val = 0.0
            if player == 0:
                ret_val = 0.0
            else:
                ret_val = -1.0
            node.update_terminal_val(game.get_encoded_board_state(), ret_val)
            return -ret_val

        if not node.is_expanded():
            # if node is not expanded, then expand the node and return the predictor state val.
            predictor_state_val = self.expand(node, game.get_encoded_board_state())
            # whoever is playing inferenced the model for state_val. we return the -ve of the val to the opponent
            return -predictor_state_val
        else:
            # if we have an expanded node, then make a decision with PUCT selection
            optimal_action = max(
                range(ACTION_SPACE),
                key=lambda a: puct_eval(
                    node.child_nodes[a].get_state_value(),
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
    def expand(self, node: Node, state):
        policy_vector, state_value = self.predictor(state)
        node.expand(state, policy_vector, state_value)
        return state_value

    def simulation(self):
        current_buffer = []
        current_node = self.root
        final_reward = 0.0
        while not done:
            player = self.game.get_current_player()
            done = self.game.is_done()
            for iter in self.MCTS_SIMS:
                game = self.game.clone()
                self.select(current_node, game, done, player)

            state = self.game.get_state()
            encoded_state = self.game.get_encoded_board_state()
            mcts_policy_vec = []
            for a in ACTION_SPACE:
                mcts_policy_vec.append(
                    current_node.child_nodes[a].get_visit_count()
                    / current_node.get_visit_count()
                )
            current_buffer.append((state, encoded_state, mcts_policy_vec))

            # use PUCT to find the next action and take a step,
            # if done then collect the final reward and update
            # the current_buffer
            optimal_action = max(
                range(ACTION_SPACE),
                key=lambda a: puct_eval(
                    current_node.child_nodes[a].get_state_value(),
                    current_node.child_nodes[a].get_prior(),
                    current_node.child_nodes[a].get_visit_count(),
                    current_node.get_visit_count(),
                ),
            )

            _, done, reward = self.game.step(optimal_action)
            current_node = current_node.child_nodes[optimal_action]
            if done:
                final_reward = reward
