import sys
import subprocess

import os


argslist = list(sys.argv)[1:]
world_size = int(argslist[0])

gpu_env = ""
for i in range(world_size):
    gpu_env += argslist[1+i]
    if (i != (world_size - 1)):
        gpu_env += ","

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env
print("Using GPU env:")
print(gpu_env)

argslist = argslist[world_size + 1:]
model_name = argslist[argslist.index('--id') + 1]

os.makedirs("log/", exist_ok=True)

gpu_log_dir = "log/" + model_name + "/"
os.makedirs(gpu_log_dir, exist_ok=True)


device_ids = None

# Add number of GPUS to args
argslist.append('--world-size')
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
    if '--gpu-rank' in argslist:
        if device_ids:
            argslist[argslist.index('--gpu-rank') + 1] = str(device_ids[i])
        else:
            argslist[argslist.index('--gpu-rank') + 1] = str(i)
    else:
        argslist.append('--gpu-rank')
        argslist.append(str(i))

    stdout = None if i == 0 else open(gpu_log_dir + "GPU_" + str(i) + ".log", "w")
    print([str(sys.executable)] + [gpu_env] + argslist)
    p = subprocess.Popen([str(sys.executable)] + ["run_deep_speech_gpu.py"] + [gpu_env] + argslist, stdout=stdout, stderr=stdout)
    workers.append(p)

try:
    for p in workers:
        p.wait()
        if p.returncode != 0:
            print(subprocess.CalledProcessError(returncode=p.returncode,
                                            cmd=p.args))
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
