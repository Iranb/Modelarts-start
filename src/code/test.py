import torch
import os
import time
os.system("nvidia-smi")
print(torch.cuda.get_device_name(0))
import time
count = 0
# a = 1 * 60 * 60 + 8000
a = 1 * 10
while (count < a):
    count_now = a - count
    print("current run time left: " ,count_now, "second")
    time.sleep(1)#sleep 1 second
    count += 1
print('done')

