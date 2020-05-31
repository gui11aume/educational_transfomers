import sys
from Bio import pairwise2

if len(sys.argv) != 3:
   print('usage: {} seqfile1 seqfile2'.format(sys.argv[0]))
   sys.exit(1)

match_score = 1
mismatch_score = 0
gap_open = 0
gap_extend = 0

with open(sys.argv[1]) as f1:
   for s1 in f1:
      s1 = s1.rstrip()
      with open(sys.argv[2]) as f2:
         for s2 in f2:
            s2 = s2.rstrip()
            a = pairwise2.align.globalms(s1.rstrip(),s2.rstrip(),match_score,mismatch_score,gap_open,gap_extend)
            print("{},{},{}".format(a[0][2], s1, s2))
