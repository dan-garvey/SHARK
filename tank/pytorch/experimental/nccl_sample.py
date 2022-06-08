import torch
from shark.shark_inference import SharkInference
import cupy as cp
from cupy.cuda import nccl
from cupy import cuda
import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)


class MulModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.train(False)

    def forward(self, x):
        return torch.mul(x, x)

@ray.remote(num_gpus=1)
def run_inference(idx, dummy_input):
    #Iree modules aren't currently pickleable, but we can loop through inputs that a read in each subprocess
    module = SharkInference(MulModule(), (dummy_input,),
                            device="gpu",
                            device_idx=idx)
    module.compile()
    #Ideally we would instead get a ptr to the cuda tensor that is the result prior to transfer back to cpu
    return module.forward((dummy_input,))


num_devices = 2
dummy_input = torch.randn(512)
recvbuf = cp.ndarray(shape=dummy_input.shape, dtype=cp.float32)
processes = []
for iters in range(1000):
    results = ray.get([run_inference.remote(str(i), dummy_input) for i in range(num_devices)])

    devs = [0, 1]
    comms = nccl.NcclCommunicator.initAll(devs)
    nccl.groupStart()
    for comm in comms:
        dev_id = comm.device_id()
        rank = comm.rank_id()
        assert rank == dev_id
        with cuda.Device(dev_id):
            sendbuf = cp.array(results[rank])
            comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, 512,
                           nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                           cuda.Stream.null.ptr)
    nccl.groupEnd()

golden = MulModule()(dummy_input)
assert (torch.allclose(torch.tensor(cp.asnumpy(recvbuf)), golden * 2))
