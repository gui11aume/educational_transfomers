import sys
import torch
import torch.nn.functional as F

from tRNA_highlighter import SeqFinder, vocab

if __name__ == '__main__':

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   model = SeqFinder(
      N = 4,             # Number of layers.
      h = 8,             # Number of attention heads.
      d_model = 256,     # Hidden dimension.
      d_ffn = 512,       # Boom dimension.
      nwrd = len(vocab)  # Input alphabet (DNA).
   )

   model.load_state_dict(torch.load(sys.argv[1]))
   model.eval()

   # Do it with CUDA if possible.
   if torch.cuda.is_available(): model.cuda()

   to_idx = lambda s: torch.LongTensor([vocab[a] for a in s]) 
   with open(sys.argv[2]) as f:
      for line in f:
         if len(line) < 50: continue
         seq = line.rstrip().upper().replace('U','T')
         fgt = [seq[i:i+200] for i in range(0,len(seq),200)]
         for seq in fgt:
            x = to_idx(seq).view(1,-1).to('cuda')
            with torch.no_grad():
               out = F.softmax(model(x), -1)
            for i in range(len(seq)):
               if out[0,i,1] < .5:
                  sys.stdout.write('%s' % seq[i].lower())
               else:
                  sys.stdout.write('%s' % seq[i].upper())
         sys.stdout.write('\n')
