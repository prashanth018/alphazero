from env.connect4 import Connect4

g = Connect4()
g.reset()

done = False
# win_trajectory = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3]
mid_game_trajectory = [0, 0, 0, 0, 0, 0, 1, 1]
for a in mid_game_trajectory:
    board, done, player = g.step(a)

# for [0, 0, 0, 0, 0, 0], this get_valid_actions should not print '0'
print("get_valid_actions", g.get_valid_actions())

print(board)
print(g.get_encoded_board_state())
print(g.is_board_full())
print(g.get_valid_actions())
print("done:", done, "winner:", player)


# tensor([[-1,  0,  0,  0,  0,  0,  0],
#         [ 1,  0,  0,  0,  0,  0,  0],
#         [-1,  0,  0,  0,  0,  0,  0],
#         [ 1,  0,  0,  0,  0,  0,  0],
#         [-1, -1, -1,  0,  0,  0,  0],
#         [ 1,  1,  1,  1,  0,  0,  0]], dtype=torch.int8)
