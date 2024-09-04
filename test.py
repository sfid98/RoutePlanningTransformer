import torch

#div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
div_term = torch.arange(0, 10, 2)
print(div_term)
