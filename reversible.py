# https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
import torch

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
   def __init__(self, F, G):
      super().__init__()
      self.F = F
      self.G = G

   def forward(self, x):
      # Compute forward pass without
      # storing the computation graph.
      x1, x2 = torch.chunk(x, 2, dim=-1)
      with torch.no_grad():
         y1 = x1 + self.F(x2)
         y2 = x2 + self.G(y1)

      return torch.cat([y1, y2], dim=-1)

   def backward(self, y, dy):
      # Compute backward pass manually using
      # the properties of reversible blocks.
      y1, y2 = torch.chunk(y, 2, dim=-1)
      dy1, dy2 = torch.chunk(dy, 2, dim=-1)
      del y, dy

      with torch.enable_grad():
         y1.requires_grad = True
         Gy1 = self.G(y1)
         torch.autograd.backward(Gy1, dy2)

      with torch.no_grad():
         x2 = y2 - Gy1
         dx1 = dy1 + y1.grad
         del y2, dy1, Gy1
         # Safety to avoid memory leak.
         y1.grad = None

      with torch.enable_grad():
         x2.requires_grad = True
         Fx2 = self.F(x2)
         # We will pass 'dx1' in output and we will need
         # to further compute derivatives with it, so
         # we need to keep the computation graph.
         torch.autograd.backward(Fx2, dx1, retain_graph=True)

      with torch.no_grad():
         x1 = y1 - Fx2
         dx2 = dy2 + x2.grad
         del y1, dy2, Fx2
         # Safety to avoid memory leak.
         x2.grad = None

         x = torch.cat([x1, x2.detach()], dim=-1)
         dx = torch.cat([dx1, dx2], dim=-1)

      return x, dx
