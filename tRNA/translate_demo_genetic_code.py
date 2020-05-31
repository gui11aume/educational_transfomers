import sys
import torch
import torch.nn.functional as F

from transformer_demo_genetic_code import *

if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   model = EncoderDecoder(
      N = 2,                    # Number of layers.
      h = 4,                    # Number of attention heads.
      d_model = 128,            # Hidden dimension.
      d_ffn = 256,              # Boom dimension.
      i_nwrd = len(DNA_vocab),  # Input alphabet (DNA).
      o_nwrd = len(prot_vocab)  # Output alphabet (protein).
   )

   model.load_state_dict(torch.load(sys.argv[1]))
   model.eval()

   # Do it with CUDA if possible.
   if torch.cuda.is_available():
      device = 'cuda'
      model.cuda()
   else:
      device = 'cpu'
   
   # Generate the data on the fly.
   DNA = random_DNA(2, 72)
   
   i_batch = to_idx(DNA, DNA_vocab).to(device)
   o_batch = torch.zeros(2,24).long().to(device)

   import pdb; pdb.set_trace()
   for _ in range(24):
      with torch.no_grad():
         z = model(i_batch, o_batch)
      probs = F.softmax(z[:,_,:], dim=-1)
      smpl = torch.distributions.Categorical(probs).sample()
      o_batch[:,_] = smpl
