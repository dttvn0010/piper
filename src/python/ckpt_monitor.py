
#export CUDA_VISIBLE_DEVICES = 

import os
import time
import sys
ckpt_path = sys.argv[1] #'/root/train/lightning_logs/version_0/checkpoints'
output_path = sys.argv[2] 
last_file = ''
while True:
   if not os.path.exists(ckpt_path):
      time.sleep(1)
      continue
   
   files = os.listdir(ckpt_path)
   if len(files) == 0 or files[0] == last_file:
      time.sleep(1)
      continue
   
   last_file = files[0]
   ep = int(last_file.split('=')[1].split('-')[0])
   if ep % 1 != 0:
      time.sleep(1)
      continue

   time.sleep(10)
   os.system(f'python3 -m piper_train.export_onnx {ckpt_path}/{last_file} {output_path}/{last_file}.onnx')
