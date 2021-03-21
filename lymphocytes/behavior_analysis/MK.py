from mkalgo.mk import mk_eab
from mkalgo.data import hospital

if __name__=='__main__':
  obj = mk_eab(l=5, metric='euclidean')
  x = hospital()
  motif_a, motif_b = obj.search(x)
  print(motif_a)
  print(motif_b)
