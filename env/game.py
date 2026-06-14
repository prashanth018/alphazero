from env.connect4 import Connect4

g = Connect4()
g.reset()

done = False
for a in [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3]:
    board, done, player = g.step(a)

print(board)
print("done:", done, "winner:", player)

# tensor([[-1,  0,  0,  0,  0,  0,  0],
#         [ 1,  0,  0,  0,  0,  0,  0],
#         [-1,  0,  0,  0,  0,  0,  0],
#         [ 1,  0,  0,  0,  0,  0,  0],
#         [-1, -1, -1,  0,  0,  0,  0],
#         [ 1,  1,  1,  1,  0,  0,  0]], dtype=torch.int8)
