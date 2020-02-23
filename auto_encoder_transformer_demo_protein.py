import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from lamb import Lamb # Local file.


'''
Relative positional encoding.
'''

def matrixR(L, d_model, ex=False):
   # Basic entries for relative positional encoding.
   inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
   if ex:
      # Returns a matrix of size (2L-1, d_model).
      sinusoid_inp = torch.ger(torch.arange(-L+1., L+0.), inv_freq)
      mat = torch.zeros(2*L-1, d_model)
      mat[:,torch.arange(0,d_model,2)] = sinusoid_inp.sin()
      mat[:,torch.arange(1,d_model,2)] = sinusoid_inp.cos()
   else:
      # Returns a matrix of size (L, d_model).
      sinusoid_inp = torch.ger(torch.arange(L+0.), inv_freq)
      mat = torch.zeros(L, d_model)
      mat[:,torch.arange(0,d_model,2)] = sinusoid_inp.sin()
      mat[:,torch.arange(1,d_model,2)] = sinusoid_inp.cos()
   return mat


'''
Dot product attention layer.
'''

class RelativeAttention(nn.Module):

   def init_matrix(self, *dims):
      m = torch.Tensor(*dims)
      # Taken from the source code of 'torch.nn.Linear'.
      torch.nn.init.kaiming_uniform_(m, a=np.sqrt(5))
      return m

   def __init__(self, h, d_model, dropout=0.1):
      assert d_model % h == 0 # Just to be sure.
      super(RelativeAttention, self).__init__()
      self.h = h
      self.d = d_model

      # Linear transformations of embeddings.
      self.Wq = nn.Parameter(self.init_matrix(d_model, d_model))
      self.Wv = nn.Parameter(self.init_matrix(d_model, d_model))
      self.We = nn.Parameter(self.init_matrix(d_model, d_model))
      self.Wr = nn.Parameter(self.init_matrix(d_model, d_model))

      # Content and position biases.
      self.cb = nn.Parameter(torch.zeros(d_model)) # Content bias.
      self.pb = nn.Parameter(torch.zeros(d_model)) # Position bias.

      # Output layers.
      self.do = nn.Dropout(p=dropout)
      self.Wo = nn.Linear(d_model, d_model)
      self.ln = nn.LayerNorm(d_model)

   def shift_rows(self, M):
      # Inspired from the Transformer-XL (but we do something different).
      # M is assumed to b a matrix of size (L1 x 2L2-1).
      N = M.shape[0]               # Batc hsize.
      h = M.shape[1]               # Number of heads.
      L1 = M.shape[-2]             # Text length (X).
      L2 = (M.shape[-1] + 1) // 2  # Text length (Y).
      if L1 == L2:
         L = L1 # = L2
      elif L1 < L2: # We need to add rows.
         L = L2
         if L % L1 == 0:
            M = M.repeat_interleave(L//L1, dim=-2)
         else:
            M = M.repeat_interleave(1+L//L1, dim=-2)
            idx = np.linspace(0, M.shape[-2]-1, L).astype(int)
            M = M[:,:,idx,:]
      elif L1 > L2: # We need to add columns.
         L = L1
         if L % L2 == 0:
            M = M.repeat_interleave(L//L2, dim=-1)
         else:
            M = M.repeat_interleave(1+L//L2, dim=-1)
            idx = np.linspace(0, M.shape[-1]-1, L).astype(int)
            M = M[:,:,:,idx]
      # M has size (L x 2L-1). Split it in two blocks of size (. x L)
      # Note: the middle column is present in both blocks.
      M1 = M[:,:,:,:L]
      M2 = M[:,:,:,L-1:]
      # Then use cat-zero-view-as-transposed-and-remove-row to shift.
      # This is a bit of a black box, but all it does is shift the rows
      # of a lower triangular and an upper triangular matrix.
      zero = torch.zeros(N,h,L,1, device=M.device, dtype=M.dtype)
      SM1 = torch.cat([zero, M1], -1).view(N,h,-1,L)[:,:,1:,:].tril(1)
      SM2 = torch.cat([M2, zero], -1).view(N,h,-1,L)[:,:,:-1,:].triu(0)
      # Then reassemble triangular matrices and we are done.
      SM = SM1 + SM2
      # Output a matrix with correct dimensions.
      if L1 == L2:
         return SM
      if L1 < L2:
         idx = np.linspace(0, L-1, L1).astype(int)
         return SM[:,:,idx,:]
      if L1 > L2:
         idx = np.linspace(0, L-1, L2).astype(int)
         return SM[:,:,:,idx]

   def forward(self, X, Y, mask=None):
      '''
            X  ~  (Batch, L1, d_model)
            Y  ~  (Batch, L2, d_model)
           W.  ~  (d_model, d_model)
       cb, pb  ~  (1, h, 1, d_model/h)
            q  ~  (Batch, h, L1, d_model/h)
          v,k  ~  (Batch, h, L2, d_model/h)
            Q  ~  (Batch, h, 2L-1, d_model/h)
            b  ~  (Batch, h, L, 2L-1)
        A,D,B  ~  (Batch, h, L, L)
           Oh  ~  (Batch, h, d_model/h, L)
            O  ~  (Batch, L, d_model)
      '''

      h  = self.h       # Number of heads.
      H  = self.d // h  # Head dimension.
      N  = X.shape[0]   # Batch size.
      L1 = X.shape[1]   # Text length (X).
      L2 = Y.shape[1]   # Text length (Y).
      L  = max(L1, L2)  # Longer text length.

      # Relative position.
      R = matrixR(L, self.d, ex=True).to(dtype=X.dtype, device=X.device)
      
      # Linear transforms.
      q = torch.matmul(X, self.Wq).view(N,L1,h,-1).transpose(1,2)
      k = torch.matmul(Y, self.We).view(N,L2,h,-1).transpose(1,2)
      v = torch.matmul(Y, self.Wv).view(N,L2,h,-1).transpose(1,2)
      # Note: Q is not the query (see p. 12 of Transformer-XL).
      Q = torch.matmul(R, self.Wr).view(1,-1,h,H).transpose(1,2)

      # Reshapes.
      pb = self.pb.view(1,h,1,-1).repeat(N,1,L1,1)
      cb = self.cb.view(1,h,1,-1).repeat(N,1,L1,1)

      # Dot products 
      B   = torch.matmul(q,  Q.transpose(-2,-1))
      D   = torch.matmul(pb, Q.transpose(-2,-1))
      A_a = torch.matmul(q,  k.transpose(-2,-1))
      A_c = torch.matmul(cb, k.transpose(-2,-1))

      # Shifted matrices (see Transformer-XL). Here we also
      # need to downsample the rows / columns because the texts
      # have different lengths.
      A_b = self.shift_rows(B)
      A_d = self.shift_rows(D)

      # Raw attention matrix.
      A = A_a + A_b + A_c + A_d

      if mask is not None:
         A = A.masked_fill(mask == 0, float('-inf'))

      # Attention softmax
      p_attn = F.softmax(A, dim=-1)

      # Apply attention to v
      Oh = torch.einsum('ijkl,ijlm->ijkm', (p_attn, v))

      # Concatenate attention output
      O = Oh.transpose(1,2).contiguous().view_as(X)

      # Layer norm and residual connection
      return self.ln(X + self.do(self.Wo(O)))


'''
Feed foward layer.
'''

class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super(FeedForwardNet, self).__init__()
      self.ff = nn.Sequential(
         nn.Linear(d_model, d_ffn),
         nn.ReLU(),
         nn.Linear(d_ffn, d_model)
      )
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, x):
      return self.ln(x + self.do(self.ff(x)))
   

''' 
Encoder and Decoder blocks.
'''

class RelativeEncoderBlock(nn.Module):
   def __init__(self, h, d_model, d_ffn, dropout=0.1):
      super(RelativeEncoderBlock, self).__init__()
      self.h = h
      self.d = d_model
      self.f = d_ffn
      self.sattn = RelativeAttention(h, d_model, dropout=dropout)
      self.ffn   = FeedForwardNet(d_model, d_ffn, dropout=dropout)
      
   def forward(self, X, mask=None):
      return self.ffn(self.sattn(X, X, mask))


class AutoEncoder(nn.Module):
   def __init__(self, N, h, d_model, d_ffn, nwrd, dropout=0.1):
      super(BERT, self).__init__()

      # Model parameters.
      self.N = N          # Number of encoders.
      self.d = d_model    # Hidden size.
      self.h = h          # Number of heads.
      self.d_ffn = d_ffn  # Boom dimension.

      # Text embedding transformations
      self.embed = nn.Embedding(nwrd, d_model)
      self.do    = nn.Dropout(p=dropout)

      # Self-attention layers (encoder)
      self.EncoderLayers = nn.ModuleList([
         RelativeEncoderBlock(h, d_model, d_ffn, dropout=dropout) \
               for _ in range(N)])

      self.mid =  nn.Sequential(
         nn.Linear(d_model, d_model),
         nn.ReLU(),
         nn.LayerNorm(d_model),
         nn.Linear(d_model, nwrd)
      )

      self.DecoderLayers = nn.ModuleList([
         RelativeEncoderBlock(h, d_model, d_ffn, dropout=dropout) \
               for _ in range(N)])

      # Final layers for reconstruction.
      self.last = nn.Sequential(
         nn.Linear(d_model, d_model),
         nn.ReLU(),
         nn.LayerNorm(d_model),
         nn.Linear(d_model, nwrd)
      )

   def forward(self, batch, mask=None):
      # Straightforward pass through the layers.
      x = self.do(self.embed(batch))
      for layer in self.EncoderLayers:
         x = layer(x)
      return self.last(x)


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

   model = BERT(
      N = 4,             # Number of layers.
      h = 8,             # Number of attention heads.
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

         batch = batch.to(device)
         rnd = lambda n: [random.randint(1,20) for _ in range(n)]

         # Choose symbols to guess (15%).
         guess_pos = torch.rand(size=batch.shape) < 0.15
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
