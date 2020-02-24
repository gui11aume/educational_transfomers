import math
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from lamb import Lamb # Local file.

'''
Single head RNN attention.
'''

class SingleHeadAttention(nn.Module):
   '''Simplified implementation of Stephen Merity's single head
   attention. See http://arxiv.org/pdf/1911.11423.pdf'''

   def __init__(self, d_model, dropout=0.1):
      super().__init__()
      self.d = d_model

      # Position-specific weights.
      self.qs = nn.Parameter(torch.zeros(d_model, dtype=torch.float))
      self.ks = nn.Parameter(torch.zeros(d_model, dtype=torch.float))
      self.vs = nn.Parameter(torch.zeros(d_model, dtype=torch.float))

      # Query projection.
      self.Q = nn.Sequential(
         nn.Linear(d_model, d_model),
         nn.LayerNorm(d_model)
      )

      # Over-parametrization.
      self.ov = nn.Linear(d_model, 2*d_model)

      # Wrap up.
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def ovrprm(self, x):
      # Cast onto the range (-1,1).
      a, b = self.ov(x).split(self.d, dim=-1)
      return torch.sigmoid(a) * torch.tanh(b)

   def forward(self, X, Y, mask=None):
      # Process input. 
      q = self.Q(X) * torch.sigmoid(self.qs)
      k = Y * torch.sigmoid(self.ks)
      v = Y * self.ovrprm(torch.sigmoid(self.vs))
      
      # Dot product attention.
      A = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d)

      if mask is not None:
         A = A.masked_fill(mask == 0, float('-inf'))

      # Attention softmax.
      p_attn = F.softmax(A, dim=-1)

      # Apply attention to v, then dropout, add
      # residuals and wrap in layer normalization.
      return self.ln(X + self.do(torch.matmul(p_attn, v)))


'''
Feed foward (boom) layer.
'''

class BoomLayer(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.d = d_model
      self.ff = nn.Sequential(
         nn.Linear(d_model, d_ffn),
         nn.ReLU(),
      )
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X):
      # Shortcut a linear transform by just summing by
      # buckets to reduce the dimension.
      Y = self.ff(X).view(*X.shape[:-1], self.d,-1).sum(dim=-1)
      return self.ln(X + self.do(Y))
   

''' 
Encoder and Decoder blocks.
'''


class EncoderAttnBlock(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.d = d_model
      self.f = d_ffn
      
      self.lstm  = nn.LSTM(d_model, d_model)
      self.ln    = nn.LayerNorm(d_model)
      self.do    = nn.Dropout(p=dropout)
      self.lnq   = nn.LayerNorm(d_model)
      self.lnk   = nn.LayerNorm(d_model)
      self.sattn = SingleHeadAttention(d_model, dropout=dropout)
      self.ffn   = BoomLayer(d_model, d_ffn, dropout=dropout)
      
   def forward(self, X, mask=None):
      X,_ = self.lstm(X)              # LSTM (no residual).
      X   = self.do(X)                # Dropout.
      X,Y = self.lnq(X), self.lnk(X)  # Split and normalize.
      X   = self.sattn(X, Y, mask)    # Self attention.
      X   = self.ffn(X)               # Boom layer.
      return X


class SHARNN(nn.Module):
   def __init__(self, N, d_model, d_ffn, nwrd, dropout=0.1):
      super().__init__()

      # Model parameters.
      self.N = N          # Number of encoders.
      self.d = d_model    # Hidden size.
      self.d_ffn = d_ffn  # Boom dimension.

      # Text embedding transformations
      self.embed = nn.Embedding(nwrd, d_model)
      self.do = nn.Dropout(p=dropout)

      # Self-attention layers
      self.EncoderLayers = nn.ModuleList([
         EncoderAttnBlock(d_model, d_ffn, dropout=dropout) \
               for _ in range(N)])

      # Final layer for reconstruction.
      self.last = nn.Linear(d_model, nwrd)

   def default_mask(self, X):
      # By default we mask the future.
      # Note: the positions marked as 0 are masked.
      L = X.shape[-2] # Text length.
      return torch.ones(L,L, device=X.device).tril()
      
   def forward(self, batch, mask=None):
      # Straightforward pass through the layers.
      X = self.do(self.embed(batch))
      # Note: masking is critical in this model.
      mask = mask if mask is not None else self.default_mask(X)
      for layer in self.EncoderLayers:
         X = layer(X, mask)
      return self.last(X)


'''
Proteins.
'''

vocab = {
   ' ':0,  'A':1,  'C':2,  'D':3,  'E':4,  'F':5,  'G':6,
   'H':7,  'I':8,  'K':9,  'L':10, 'M':11, 'N':12, 'P':13,
   'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20,
   '_':21, '*':22, # '_' = STOP, '*' = MASK.
}


class SeqData:

   def __init__(self, path, vocab):
      self.vocab = vocab
      # Remove lines with unknown characters.
      is_clean = lambda s: set(s.rstrip()).issubset(vocab)
      with open(path) as f:
         self.data = [line.rstrip() for line in f if is_clean(line)]

   def batches(self, btchsz=32, randomize=True):
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(self.data))
      if randomize: np.random.shuffle(idx)
      to_idx = lambda s: torch.LongTensor([self.vocab[a] for a in s])
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         data = [to_idx(self.data[i]) for i in ix]
         yield torch.nn.utils.rnn.pad_sequence(data, batch_first=True)


if __name__ == "__main__":

   model = SHARNN(
      N = 4,             # Number of layers.
      d_model = 256,     # Hidden dimension.
      d_ffn = 512,       # Boom dimension.
      nwrd = len(vocab)  # Output alphabet (protein).
   )

   protdata = SeqData('proteins.txt', vocab)
   mask_symbol = len(vocab)-1 # The last symbol is the mask.

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda': model.cuda()
   
   lr  = 0.001 # The celebrated learning rate.
   per = 512  # Half period of the cyclic learning rate.

   # Optimizer (warmup and linear decay or LR)
   opt = Lamb(model.parameters(),
         lr=lr, weight_decay=0.01, betas=(.9, .999), adam=True)

   loss_fun = nn.CrossEntropyLoss(reduction='mean')
   lrval = list(range(per)) + list(range(per,0,-1))

   nbtch = 0
   for epoch in range(20):
      epoch_loss = 0.
      for batch in protdata.batches():
         nbtch += 1
         # Change the learning rate (cycles).
         opt.param_groups[0]['lr'] = lr * lrval[nbtch % (2*per)] / per

         rnd = lambda n: [random.randint(1,20) for _ in range(n)]

         # Choose symbols to guess (15%).
         guess_pos = (torch.rand(batch.shape) < 0.15) & (batch > 0)
         # Record original symbols (targets).
         batch = batch.to(device)
         trgt = batch[guess_pos].clone()
         # BERT masked language model (MLM) protcol:
         # 80% mask, 10% random, 10% unchanged.
         rnd_pos = guess_pos & (torch.rand(size=batch.shape) < 0.5)
         msk_pos = guess_pos & (torch.rand(size=batch.shape) < 0.8)
         batch[rnd_pos] = torch.tensor(rnd(torch.sum(rnd_pos))).to(device)
         batch[msk_pos] = mask_symbol

         z = model(batch)
         loss = loss_fun(z[guess_pos], trgt)

         # Update.
         opt.zero_grad()
         loss.backward()
         opt.step()

         epoch_loss += float(loss)

      sys.stderr.write('Epoch %d, loss: %f\n' % (epoch+1, epoch_loss))
      if (epoch+1) % 10 == 0:
         torch.save(model.state_dict(), 'model_epoch_%d.tch' % (epoch+1))
