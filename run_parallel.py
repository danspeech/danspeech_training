"""
Note that this is the file we use at DanSpeech for multi GPU training.

You probably need to modify for your own GPU setup.
"""
import sys
import subprocess

import os

argslist = list(sys.argv)[1:]
world_size = int(argslist[1])
sub_file = argslist[0]

print(sub_file)

gpu_env = ""
for i in range(world_size):
    gpu_env += argslist[2 + i]
    if (i != (world_size - 1)):
        gpu_env += ","

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env
print("Using GPU env:")
print(gpu_env)

argslist = argslist[world_size + 2:]
model_name = argslist[argslist.index('--model_id') + 1]

print(argslist)

os.makedirs("log/", exist_ok=True)

gpu_log_dir = "log/" + model_name + "/"
os.makedirs(gpu_log_dir, exist_ok=True)

device_ids = None

# Add number of GPUS to args
argslist.append('--world_size')
argslist.append(str(world_size))

workers = []


for i in range(world_size):

    # Create rank
    if '--rank' in argslist:
        argslist[argslist.index('--rank') + 1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))

    # Create gpu_rank
    if '--gpu_rank' in argslist:
        argslist[argslist.index('--gpu_rank') + 1] = str(i)
    else:
        argslist.append('--gpu_rank')
        argslist.append(str(i))

    stdout = None if i == 0 else open(gpu_log_dir + "GPU_" + str(i) + ".log", "w")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_env
    print([str(sys.executable)] + argslist)
    p = subprocess.Popen([str(sys.executable)] + ["{}.py".format(sub_file)] + argslist, stdout=stdout,
                         stderr=stdout)
    workers.append(p)

try:
    for p in workers:
        p.wait()
        if p.returncode != 0:
            print(subprocess.CalledProcessError(returncode=p.returncode, cmd=p.args))
            break

    for p in workers:
        p.terminate()
        p.wait()


except KeyboardInterrupt:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for p in workers:
        p.terminate()
        p.wait()
    exit()
