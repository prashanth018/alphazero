# AlphaZero

## Implementation:
- env: connect4

Below are some of the properties of Connect4 that is worth noting for AlphaZero design
- "Is the state Markovian?": Frame stacking is not needed in Connect 4. Current state (i.e., `util.encode_board_state`) fully represents all the historic context needed to make the optimal next move. 
    - "Why did AlphaZero have to stack frames in Chess? Don't we solve puzzles on chess.com without having to know the trajectory?": For tactical positions, yes. But moves like Castling, En passant, repetition rules for draw break the pure markovian property. Therefore, 8 past positions and extra planes for castling rights, repetition etc.,
- Fully observable.


How is MCTS node evaluation different from Minimax?
Intuition: Every Player evaluates their position from their perspective. Assuming that its Player 1's turn to play, they play to maximize their return. A detour; Connect 4 is a zero-sum game. Therefore, "good for me = bad for you". Now, when player 1 makes a move and its player 2's turn, they play to maximize their position. Their state (which is a tensor(2,6,7)) already represents their position (plane 0 is player 2's pieces and plane 1 is player 1's pieces). Now assuming Player 2's state value was V(P2|action), then Player 1's state value would be `- min of all actions (V(P2|action))`. Now imagine the game is won by Player 1. Player 1 gets a reward of +1 on his turn. The replay buffer entry for Player 1's would look like: `tuple(tensor(plane0:player1, plane1:player2), policy_vector, +1)`. Buffer entry for Player 2's turn would look like: `tuple(tensor(plane0:player2, plane1:player1), policy_vector, -1)` since the reward sign gets flipped.


Why tanh and not sigmoid?
To be able to express -1 to +1 range for state values. Sigmoid represents 0 to +1

Why no softmax inside the model?
Similar to Transformers, we leave out the normalization to training step. This is so that we implement custom mask logic (based on SFT/training etc.,). In this case we mask the illegal moves to -inf. Therefore, we leave the policy_outputs as raw logits and instead leave the masking and softmax logic to loss functions.

Questions:
- How is AlphaZero not highly customized for the game? Also, AlphaZero algorithm seems to be only for zero-sum game? - can it also be applied for collaborative games? 