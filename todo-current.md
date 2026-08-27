# AlphaZero — current work (pick up here)

> Active implementation TODOs for the AlphaZero build itself.
> (Post-project backlog lives in [to-do.md](to-do.md).)

## MCTS / self-play loop
- [x] Use PUCT to select the next action and take a step; if terminal, collect the final reward and update `current_buffer`
- [x] Verify the tree is **not** destroyed after each iteration
- [ ] Prune the sister subtrees (reuse the child of the played move as the new root)
- [x] Verify terminal conditions and edge cases
- [x] Verify node stat values are populated correctly (N, W, Q, P)
- [x] Simplify the `Node` class after debugging
- [x] Handle illegal moves by filtering them & renormalizing the distribution in PUCT & expand node
- [ ] Sample from distribution instead of PUCT while actually making a move
- [ ] Add temperature schedule for sampling the next move in an actual game
- [ ] Add Root Dirichlet noise

## Game env
- [x] Implement `clone()` method on the game class
- [x] Implement `get_decoded_board_state`

## Network / predictor
- [ ] Predictor policy vector needs softmaxing
- [ ] Wire the real net into `expand` (replace `get_mock_predictor_val`) — policy logits + value head (tanh)
- [ ] Confirm net input matches `get_encoded_board_state` (2, 6, 7)

## Training loop
- [ ] Loss = value MSE + policy cross-entropy (+ L2 reg)
- [ ] Train step: sample minibatch from replay buffer → forward → loss → backprop → step
- [ ] Optimizer + LR schedule
- [ ] Persistent replay buffer that accumulates across games (FIFO cap) — not the per-game `current_buffer`
- [ ] Outer loop: self-play → add to buffer → train → (eval/gate) → repeat
- [ ] Checkpointing — save/load model (+ optimizer + step)
- [ ] Data augmentation — mirror board & policy (Connect4 left-right symmetry)

## Evals
- [ ] Add evals against a random agent/perfect solver every few epochs
- [ ] (Optional) Gating — only promote a new net if it beats the current best

## Intuition
- [ ] Understand intuition behind each of the constants