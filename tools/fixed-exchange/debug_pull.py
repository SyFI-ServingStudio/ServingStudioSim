import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world):
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    from profiling.runners.comm.mnnvl_fixed_exchange import FixedExchange
    ex = FixedExchange(rank, world, 16, 4096, 8)
    ex.set_counts([2]*world)
    for s in (0, 1):
        p = ex.partials(s, 2*world)
        p.copy_(torch.arange(2*world, device="cuda", dtype=torch.float32)[:, None].expand(-1, 4096).to(torch.bfloat16) + 100*(rank+1) + 1000*s)
    torch.cuda.synchronize(); dist.barrier()
    st = ex.local[:64].view(torch.int32).clone()
    from flashinfer.comm.mnnvl import create_tensor_from_cuda_memory
    off = ex.layout.partials
    if rank == 0:
        print("layout", ex.layout, "slices", [hex(x) for x in ex.slices], flush=True)
        for peer in range(world):
            v = create_tensor_from_cuda_memory(ex.slices[peer] + off, (8,), torch.bfloat16, rank)
            print("peer", peer, v.tolist(), flush=True)
    out = torch.zeros((2, 4096), dtype=torch.bfloat16, device="cuda")
    ex.pull_combine(out); torch.cuda.synchronize()
    st2 = ex.local[:128].view(torch.int32).clone()
    if rank in (0, 1):
        print(rank, "state before", st.tolist()[:8], "after", st2.tolist(), "out", out[:, :4].tolist(), "expect row", 2*rank, flush=True)
    ex.close(); dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]="29611"
    mp.spawn(worker, args=(8,), nprocs=8, join=True)
