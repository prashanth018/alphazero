# AlphaZero

## Implementation:
- env: connect4

Below are some of the properties of Connect4 that is worth noting for AlphaZero design
- "Is the state Markovian?": Frame stacking is not needed in Connect 4. Current state (i.e., `util.encode_board_state`) fully represents all the historic context needed to make the optimal next move. 
    - "Why did AlphaZero have to stack frames in Chess? Don't we solve puzzles on chess.com without having to know the trajectory?": For tactical positions, yes. But moves like Castling, En passant, repetition rules for draw break the pure markovian property. Therefore, 8 past positions and extra planes for castling rights, repetition etc.,
- Fully observable.
- Given that the board game environment is deterministic, Q(s,a) = V(s') (transition prob is 1).


How is MCTS node evaluation different from Minimax?
Intuition: Every Player evaluates their position from their perspective. Assuming that its Player 1's turn to play, they play to maximize their return. A detour; Connect 4 is a zero-sum game. Therefore, "good for me = bad for you". Now, when player 1 makes a move and its player 2's turn, they play to maximize their position. Their state (which is a tensor(2,6,7)) already represents their position (plane 0 is player 2's pieces and plane 1 is player 1's pieces). Now assuming Player 2's state value was V(P2|action), then Player 1's state value would be `- min of all actions (V(P2|action))`. Now imagine the game is won by Player 1. Player 1 gets a reward of +1 on his turn. The replay buffer entry for Player 1's would look like: `tuple(tensor(plane0:player1, plane1:player2), policy_vector, +1)`. Buffer entry for Player 2's turn would look like: `tuple(tensor(plane0:player2, plane1:player1), policy_vector, -1)` since the reward sign gets flipped.

Dirichlet Distribution & Noise:
- `from torch.distributions.dirichlet import Dirichlet; dist = Dirichlet(alphas); sample = dist.sample()`
- Properties:
    - Dirichlet Distribution helps generate noise for a multivariate distribution. It depends on the initial vector called `alpha`. If `alpha` vector elems are too small like `tensor([0.1, 0.1, 0.1])`, then the outcome is too sharp and equally possible with 0,1 or 2. For example:
        ```
        >>> o = torch.ones((3)) * 0.1
        >>> Dirichlet(o).sample()
        tensor([4.8086e-01, 1.4877e-07, 5.1914e-01])
        >>> Dirichlet(o).sample()
        tensor([9.9322e-01, 6.7825e-03, 5.4205e-09])
        ``` 
    - 

Why tanh and not sigmoid?
To be able to express -1 to +1 range for state values. Sigmoid represents 0 to +1

Why no softmax inside the model?
Similar to Transformers, we leave out the normalization to training step. This is so that we implement custom mask logic (based on SFT/training etc.,). In this case we mask the illegal moves to -inf. Therefore, we leave the policy_outputs as raw logits and instead leave the masking and softmax logic to loss functions.

How does rollouts get away with predictor returning bad priors?
The magic happens in the PUCT formula. As a specific bad move gets visited, its $N(s,a)$ goes up, which forces its exploration term down. Even if the network gives that move a massive initial prior $P(s,a) = 0.90$, a few bad rollouts will tank its $Q(s,a)$ value. The MCTS effectively "overrules" the network's high prior by refusing to visit that node anymore.

Consider example:  s (take action a) - s1 (take action a1) - s2; and s2 is expanded for the first time. s2 returns you a predictor_state_value. Why do we not use the prior to weight update the V(s1) during the backup?
Prior P is only used in PUCT selection. 
- So when v(s2) comes back:
    - s2 (just expanded): N=1, W=v2.
    - s1: s1.N += 1, s1.W += (−v2) → Q(s1) = W/N. (sign-flipped once because s1 is the opponent's turn relative to s2)
    - s: s.N += 1, s.W += (+v2) → sign flips again.
Prior steers where simulations go (via selection); backup just averages what comes back (with a sign flip per level)
Intuition: The whole idea behind MCTS is to search for the paths to success and stamp those paths with +1 and -1s - which will be our ground truth. We don't want network's raw opinion to contaminate the averages.

What does MCTS do for invalid actions?
Irrespective of what the ResNet returns, we always 0 out the probability for invalid actions and then renormalize the probability. We also do not create child nodes for invalid actions. Now, given that we don't create a child node, optimal action selection would exclude invalid moves during PUCT. Why do this? The goal of MCTS is to generate GT policy vectors and state values. By forcing MCTS to not take the invalid moves, we force the probability of invalid actions to 0. This enables NN to learn the rules & boundaries of the game by itself.

Why not use the state values to populate the buffer instead of using terminal game outcome?
The state/action values in MCTS are computed by averaging the evaluations of the leaf nodes. Leaf node values are computed by the ResNet. Using these values to retrain the ResNet would creates a feedback loop known as bootstrapping bias. Therefore by training strictly on 'z' we break the loop and ground to the reality.   

Important Note:
Realized a bug. Below was how I coded the PUCT selection initially and the tree ended up actively ignoring the best move.
```
optimal_action = max(
    valid_actions,
    key=lambda a: puct_eval(
        current_node.child_nodes[a].get_state_value(),
        current_node.child_nodes[a].get_prior(),
        current_node.child_nodes[a].get_visit_count(),
        current_node.get_visit_count(),
    ),
)
```
This is a move being played from the node's perspective. We either minimize child nodes' values and then use PUCT or we maximize the -ve of the child nodes' state values. We essentially, negamax to choose the right action. Below is the right logic.
```
optimal_action = max(
    valid_actions,
    key=lambda a: puct_eval(
        -current_node.child_nodes[a].get_state_value(),
        current_node.child_nodes[a].get_prior(),
        current_node.child_nodes[a].get_visit_count(),
        current_node.get_visit_count(),
    ),
)
```

### Learning
- L1 Norm: `from torch.nn.functional import normalize; normalize(p_v, p=1, dim=0)`
- L2 Norm: `from torch.nn.functional import normalize; normalize(p_v, p=2, dim=0)`
- L1 norm is manhattan distance i.e., `abs(p0) + abs(p1) + ..` while L2 norm is euclidean distance i.e., `sqrt(p0**2 + p1**2 + ..)`
- To get distance, euclidean distance = `torch.linalg.vector_norm(v, ord=2)`; manhattan distance = `torch.linalg.vector_norm(v, ord=1)`


Questions:
- How is AlphaZero not highly customized for the game? Also, AlphaZero algorithm seems to be only for zero-sum game? - can it also be applied for collaborative games?
- Why are we not penalizing the net for predicting the illegal moves? The approach seems to be more like "teach the network what the right to do is not forget about the wrong things - like edge cases".