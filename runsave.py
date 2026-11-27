import torch

checkpoint = torch.load("rescued_checkpoint.pth", map_location="cpu")

print(checkpoint["epoch"])
print(checkpoint["loss"])
