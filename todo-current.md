# AlphaZero — current work (pick up here)

> Active implementation TODOs for the AlphaZero build itself.
> (Post-project backlog lives in [to-do.md](to-do.md).)

## MCTS / self-play loop
- [ ] Use PUCT to select the next action and take a step; if terminal, collect the final reward and update `current_buffer`
- [ ] Verify the tree is **not** destroyed after each iteration — ask how to prune the sister subtrees (reuse the child of the played move as the new root)
- [ ] Verify terminal conditions and edge cases
- [ ] Verify node stat values are populated correctly (N, W, Q, P)
- [ ] Simplify the `Node` class after debugging

## Game env
- [/] Implement `clone()` method on the game class
- [/] Implement `get_decoded_board_state`

## Network / predictor
- [ ] Predictor policy vector needs softmaxing
