import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=1, batch_first=True)
# data = torch.randn(2, 2, 3)
data = [[[1, 3, 4],
         [4, 5, 6]],
        [[6, 7, 8]]]
data = torch.tensor(data)
output, (h, c) = lstm(data)

print(output)
print(h)
print(c)


