import torch
import time

bla = torch.randn(10, 100, 200)

reshape_time = 0
for i in range(1000):
    start = time.time()
    blub = bla.clone()
    blub.reshape(blub.shape[0], -1)
    reshape_time += time.time() - start

view_time = 0
for i in range(1000):
    start = time.time()
    blub = bla.clone()
    blub.view(blub.shape[0], -1)
    view_time += time.time() - start



print(f'View time: {view_time}')
print(f'Reshape time: {reshape_time}')