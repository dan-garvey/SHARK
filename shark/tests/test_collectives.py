import cupy as cp
import psutil
from shark.collective import collective, types
import ray

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)


@ray.remote(num_gpus=2, max_calls=1)
def test_reduce_multigpu():
    cp.cuda.Device(0).use()
    a = cp.ones(1)

    cp.cuda.Device(1).use()
    b = cp.zeros(1)

    c = collective.reduce_multigpu([a,b])
    assert cp.equal(a,c)
    

@ray.remote(num_gpus=2, max_calls=1)
def test_reducescatter_multigpu():
    cp.cuda.Device(0).use()
    a = cp.ones(1)

    cp.cuda.Device(1).use()
    b = cp.zeros(1)

    c = collective.reducescatter_multigpu([a,b])
    assert cp.equal(c,[a,b])
    
@ray.remote(num_gpus=2, max_calls=1)
def test_allgather_multigpu():
    cp.cuda.Device(0).use()
    a = cp.ones(1)

    cp.cuda.Device(1).use()
    b = cp.zeros(1)

    c = collective.allgather_multigpu([a,b])
    assert cp.equal(c,[a,b])
    

@ray.remote(num_gpus=2, max_calls=1)
def test_broadcast_multigpu():
    cp.cuda.Device(0).use()
    a = cp.ones(1)

    cp.cuda.Device(1).use()
    b = cp.zeros(1)

    c = collective.broadcast_multigpu([a,b])
    assert cp.equal(c,a.concatenate(b,0))
    



if __name__=="__main__":
    test_reduce_multigpu.remote()

