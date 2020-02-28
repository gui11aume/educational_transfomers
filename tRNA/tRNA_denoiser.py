import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers import Lamb, Lookahead # Local file.


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
      super().__init__()
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

      # Relative position.
      R = matrixR(L2, self.d, ex=True).to(dtype=X.dtype, device=X.device)
      
      # Linear transforms.
      q = torch.matmul(X, self.Wq).view(N,L1,h,-1).transpose(1,2)
      k = torch.matmul(Y, self.We).view(N,L2,h,-1).transpose(1,2)
      v = torch.matmul(Y, self.Wv).view(N,L2,h,-1).transpose(1,2)
      # Note: Q is not the query (see p. 12 of Transformer-XL).
      Q = torch.matmul(R, self.Wr).view(1,-1,h,H).transpose(1,2)

      # Reshapes.
      pb = self.pb.view(1,h,1,-1).repeat(N,1,L1,1)
      cb = self.cb.view(1,h,1,-1).repeat(N,1,L1,1)

      # Dot products.
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

      # Attention softmax.
      p_attn = F.softmax(A, dim=-1)

      # Apply attention to v.
      Oh = torch.matmul(p_attn, v)

      # Concatenate attention output.
      O = Oh.transpose(1,2).contiguous().view_as(X)

      # Layer norm and residual connection.
      return self.ln(X + self.do(self.Wo(O)))


'''
Feed foward layer.
'''

class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.ff = nn.Sequential(
         nn.Linear(d_model, d_ffn),
         nn.ReLU(),
         nn.Linear(d_ffn, d_model)
      )
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X):
      return self.ln(X + self.do(self.ff(X)))


''' 
Encoder and Decoder blocks.
'''

class RelativeEncoderBlock(nn.Module):
   def __init__(self, h, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.h = h
      self.d = d_model
      self.f = d_ffn
      self.sattn = RelativeAttention(h, d_model, dropout=dropout)
      self.ffn   = FeedForwardNet(d_model, d_ffn, dropout=dropout)
      
   def forward(self, X, mask=None):
      return self.ffn(self.sattn(X, X, mask))


class RelativeDecoderBlock(nn.Module):
   def __init__(self, h, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.h = h
      self.d = d_model
      self.f = d_ffn
      self.sattn = RelativeAttention(h, d_model, dropout=dropout)
      self.oattn = RelativeAttention(h, d_model, dropout=dropout)
      self.ffn   = FeedForwardNet(d_model, d_ffn, dropout=dropout)

   def default_mask(self, X):
      # By default we mask the future.
      # Note: the positions marked as 0 are masked.
      L = X.shape[-2] # Text length.
      return torch.ones(L,L, device=X.device).tril()
      
   def forward(self, X, Y, mask=None):
      mask = mask if mask is not None else self.default_mask(X)
      X = self.sattn(X, X, mask)  # Self (masked) attention.
      X = self.oattn(X, Y)        # Other (unmasked) attention.
      return self.ffn(X)


class EncoderDecoder(nn.Module):
   def __init__(self, N, h, d_model, d_ffn, i_nwrd, o_nwrd, dropout=0.1):
      super().__init__()

      # Model parameters.
      self.N = N          # Number of encoders.
      self.d = d_model    # Hidden size.
      self.h = h          # Number of heads.
      self.d_ffn = d_ffn  # Boom dimension.

      # Text embedding transformations.
      self.i_emb = nn.Embedding(i_nwrd, d_model)
      self.o_emb = nn.Embedding(o_nwrd, d_model)
      self.do    = nn.Dropout(p=dropout)

      # Self-attention layers of the encoder.
      self.EncoderLayers = nn.ModuleList([
         RelativeEncoderBlock(h, d_model, d_ffn, dropout=dropout) \
               for _ in range(N)])
      # Self/other attention layers of the decoder.
      self.DecoderLayers = nn.ModuleList([
         RelativeDecoderBlock(h, d_model, d_ffn, dropout=dropout) \
               for _ in range(N)])

      # Class head for pre-training.
      self.cls = nn.Linear(d_model, 13)

      # Final layer (decoder side) for reconstruction.
      self.last = nn.Linear(d_model, o_nwrd)

   def shift(self, batch):
      # Prepend symbol 0 and shift sequences right.
      N = batch.shape[0] # Batch size.
      start = torch.zeros(N,1).long().to(batch.device)
      return torch.cat([start, batch[:,:-1]], dim=-1)

   def forward(self, i_batch, o_batch=None, mask=None, pretrain=False):
      # Pretrain mode: guess the class.
      if pretrain:
         i_embeddings = self.i_emb(i_batch)
         i = self.do(i_embeddings)
         # Attention layers (encoder).
         for layer in self.EncoderLayers:
            i = layer(i)
         return self.cls(i)
      # Fine-tuning mode: translate.
      else:
         i_embeddings = self.i_emb(i_batch)
         o_embeddings = self.o_emb(self.shift(o_batch))

         i = self.do(i_embeddings)
         o = self.do(o_embeddings)

         # Attention layers (encoder).
         for layer in self.EncoderLayers:
            i = layer(i)
         # Attention layers (decoder).
         for layer in self.DecoderLayers:
            o = layer(o,i)

         return self.last(o)


'''
DNA.
'''

vocab = {
   ' ':0, 'A':1, 'C':2, 'G':3, 'T':4,
   'a':5, 'c':6, 'g':7, 't':8,
   # Special symbols
   'START':9, 'CLS':10,
   # Sequence symbols.
   11:11, 12:12, 13:13, 14:14, 15:15,
   16:16, 17:17, 18:18, 19:19, 20:20,
   21:21, 22:22,
}

class SeqData:

   def __init__(self, path, vocab):
      self.vocab = vocab
      # Remove lines with unknown characters.
      is_clean = lambda s: set(s.rstrip()).issubset(vocab)
      with open(path) as f:
         self.data = [line.rstrip() for line in f if is_clean(line)]

   def mutf(self, oseq, pm=.15, pd=.15, pi=.15, outlen=200):
      '''Mutations, deletions and insertions set to 0-0.15 by default.'''
      # Draw probabilities at random.
      pm = pm *random.random()
      pd = pd *random.random()
      pi = pi *random.random()
      # Deletions.
      seq = [a for a in oseq if random.random() > pd]
      # Mutations.
      M = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
      mut = lambda a: a if random.random() > pm else random.choice(M[a])
      seq = [mut(a) for a in seq]
      # Insertions (only 1 nucleotide).
      I = 'ACGT'
      ins = lambda a: a if random.random() > pi else a + random.choice(I)
      seq = [ins(a) for a in seq]
      # Done.
      return ''.join(seq)

   def batches(self, pm=.15, pd=.15, pi=.15, btchsz=64):
      '''Mutations, deletions and insertions set to 0-0.15 by default.'''
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(self.data))
      to_idx = lambda s: torch.LongTensor([self.vocab[a] for a in s])
      mess = lambda s: ''.join(random.choice('GATC') for _ in s)
      # Define a generator for convenience.
      for _ in range(512):
         seqid = random.choices(range(len(self.data)), k=btchsz)
         o_seq = [[s+11] for s in seqid]
         i_seq = [self.mutf(self.data[s], pm, pd, pi) for s in seqid]
         junks = [mess(self.data[s]) for s in seqid] # Junk stuff.
         i_seq = [to_idx(s) for s in i_seq + junks]
         o_seq = [to_idx(s) for s in o_seq + junks]
         i = torch.nn.utils.rnn.pad_sequence(i_seq, batch_first=True)
         o = torch.nn.utils.rnn.pad_sequence(o_seq, batch_first=True)
         seqid = torch.LongTensor(seqid + [12] * len(seqid))
         yield i,o,seqid


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   model = EncoderDecoder(
      N = 4,                # Number of layers.
      h = 8,                # Number of attention heads.
      d_model = 256,        # Hidden dimension.
      d_ffn = 512,          # Boom dimension.
      i_nwrd = len(vocab),  # Input alphabet (DNA).
      o_nwrd = len(vocab)   # Output alphabet (DNA).
   )

   tRNAdata = SeqData('yeast_tRNA.txt', vocab)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda': model.cuda()
   
   lr  = 0.0001 # The celebrated learning rate.

   # Optimizer (warmup and linear decay or LR)
   baseopt = Lamb(model.parameters(),
         lr=lr, weight_decay=0.01, betas=(.9, .999), adam=True)
   opt = Lookahead(base_optimizer=baseopt, k=5, alpha=0.8)


   # Pre-training (5 epochs).
   clweight = torch.FloatTensor([4,4,4,4,4,4,4,4,4,4,4,4,1]).to(device)
   loss_fun = nn.CrossEntropyLoss(weight=clweight, reduction='mean')

   nbtch = 0
   for epoch in range(5):
      epoch_loss = 0.
      for batch in tRNAdata.batches():
         nbtch += 1

         # Text to classify.
         i_batch = batch[0].to(device)
         cls = batch[2].to(device)

         # Prepend 'CLS' token.
         tok = torch.LongTensor([vocab['CLS']]).to(device)
         i_batch = torch.cat([tok.repeat(i_batch.shape[0],1), i_batch], 1)

         z = model(i_batch, pretrain=True)
         loss = loss_fun(z[:,0,:].view(-1,13), cls)

         # Update.
         opt.zero_grad()
         loss.backward()
         opt.step()

         epoch_loss += float(loss)

      sys.stderr.write('Epoch %d, loss: %f\n' % (epoch+1, epoch_loss))
      if (epoch+1) % 5 == 0:
         torch.save(model.state_dict(), 'pre_trained_model_epoch_%d.tch' % (epoch+1))

   # Fine-tuning.
   clweight = torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,
      50,50,50,50,50,50,50,50,50,50,50,50]).to(device)
   loss_fun = nn.CrossEntropyLoss(weight=clweight, reduction='mean')

   nbtch = 0
   for epoch in range(50):
      epoch_loss = 0.
      for batch in tRNAdata.batches():
         nbtch += 1

         # Text to denoise.
         i_batch = batch[0].to(device)
         o_batch = batch[1].to(device)

         import pdb; pdb.set_trace()
         z = model(i_batch, o_batch)
         loss = loss_fun(z.view(-1,len(vocab)), o_batch.view(-1))

         # Update.
         opt.zero_grad()
         loss.backward()
         opt.step()

         epoch_loss += float(loss)

      sys.stderr.write('Epoch %d, loss: %f\n' % (epoch+1, epoch_loss))
      if (epoch+1) % 10 == 0:
         torch.save(model.state_dict(), 'model_epoch_%d.tch' % (epoch+1))
