import sys
import torch
import torch.nn.functional as F

from tRNA_denoiser import *

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

   model.load_state_dict(torch.load(sys.argv[1]))
   tRNAdata = SeqData('yeast_tRNA.txt', vocab)

   # Do it with CUDA if possible.
   if torch.cuda.is_available():
      device = 'cuda'
      model.cuda()
   else:
      device = 'cpu'
   
   model.eval()
   # Generate the data on the fly.
   for batch in tRNAdata.batches():
      # Text to denoise.
      i_batch = batch[0].to(device)
      o_batch = batch[1].to(device)
      o_batch[:,:] = 0
   
      for _ in range(o_batch.shape[1]):
         with torch.no_grad():
            z = model(i_batch, o_batch)
         probs = F.softmax(z[:,_,:], dim=-1)
         smpl = torch.distributions.Categorical(probs).sample()
         o_batch[:,_] = smpl
      # Print sequences.
      toseq = lambda t: ''.join([' ACGTacgt![0123456789+*'[x] for x in t])
      for _ in range(batch[0].shape[0]):
         print(toseq(i_batch[_,:]))
         print(toseq(o_batch[_,:]))
         print(toseq(batch[1][_,:]))
         print('---')
      break
