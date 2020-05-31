import sys
import torch
import torch.nn.functional as F

from tRNA_capitalizer import EncoderDecoder, vocab

if __name__ == '__main__':

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   model = EncoderDecoder(
      N = 4,               # Number of layers.
      h = 8,               # Number of attention heads.
      d_model = 256,       # Hidden dimension.
      d_ffn = 512,         # Boom dimension.
      i_nwrd = len(vocab), # Input alphabet (DNA).
      o_nwrd = len(vocab)  # Input alphabet (DNA).
   )

   model.load_state_dict(torch.load('model_epoch_50.tch'))
   model.eval()

   # Do it with CUDA if possible.
   if torch.cuda.is_available(): model.cuda()

   to_idx = lambda s: torch.LongTensor([vocab[a] for a in s]) 
   with open(sys.argv[1]) as f:
      for line in f:
         if len(line) < 50: continue
         seq = line.rstrip().lower().replace('U','T')
         fgt = [seq[i:i+200] for i in range(0,len(seq),50)]
         for seq in fgt:
            if len(seq) < 50: break
            i_batch = to_idx(seq).view(1,-1).to('cuda')
            o_batch = i_batch.clone()
            o_batch[:,:] = 0
            for _ in range(o_batch.shape[1]):
               with torch.no_grad():
                  z = model(i_batch, o_batch)
               probs = F.softmax(z[:,_,:], dim=-1)
               smpl = torch.distributions.Categorical(probs).sample()
               o_batch[:,_] = smpl
            # Print sequences.
            toseq = lambda t: ''.join([' ACGTacgt!'[x] for x in t])
            print(toseq(i_batch[0,:]))
            print(toseq(o_batch[0,:]))
            print('---')
