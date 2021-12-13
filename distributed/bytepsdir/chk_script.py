import torch
import torch.optim
import torchvision.models as models
import time
from collections import OrderedDict
import os

def _to_cpu(ele, snapshot=None):
   #while True:
   if snapshot is None:
      snapshot = {}

   if hasattr(ele, 'cpu'):
      snapshot = ele.cpu()
   elif isinstance(ele, dict):
      snapshot = {}
      for k,v in ele.items():
          snapshot[k] = None
          snapshot[k] = _to_cpu(v, snapshot[k])
   elif isinstance(ele, list):
          snapshot  = [None for _ in range(len(ele))]
          for idx, v in enumerate(ele):
              snapshot[idx] = _to_cpu(v, snapshot[idx])
   else:
          return ele
   return snapshot	


def snap_and_persist(model, optimizer, i):
   
   state = {'model': model, 'optimizer': optimizer}
   start = time.time()
   latest_snapshot = OrderedDict()
   # Snapshot the state of tractable items
   for name, ref in state.items():
       if name not in latest_snapshot:
           if hasattr(ref, 'state_dict'):
               latest_snapshot[name] = _to_cpu(ref.state_dict())
           else:
               latest_snapshot[name] = {}
               for n,r in ref.items():
                   latest_snapshot[name][n] = _to_cpu(r)				
   end = time.time()
   ts = end-start

   fp = 'check' + str(i) + '.chk'
   start = time.time()
   torch.save(latest_snapshot, fp)
    
   f = open(fp, 'a+')
   os.fsync(f.fileno())
   f.close()

   tp = time.time() - start 
   return ts, tp



def main():

   torch.cuda.set_device(0)
   arch = 'resnet50'
   classes = 1000
   model = models.__dict__[arch](num_classes=classes)
   model = model.cuda()

   # just an SGD optimizer
   optimizer = torch.optim.SGD(model.parameters(), 0.1)
   ts = []
   tp = []

   # warm-up
   for i in range(5):
     snap_and_persist(model, optimizer, 100+i)
  
   for i in range(100):
     stime, ptime = snap_and_persist(model, optimizer, i)
     ts.append(stime)
     tp.append(ptime)
     ts.sort()
     tp.sort()
   
   print("--- snapshot: ", ts[50], ts[95])
   print('--- persist: ', tp[50], tp[95])


if __name__ == '__main__':
    main()
