"""PyNCCL rank groups that span nodes (``fabric='infiniband'``).

An EP collective over InfiniBand needs ranks on more than one node. The remote
job that runs ``kernel-profile`` on a multi-node pyxis allocation already starts
a peer agent on every node but the first (`alignment.profiler.multinode peer`);
this launcher drives those agents through the same shared-directory protocol a
multi-node capture uses:

1. The head writes the rank group's spec (the per-rank function, its kwargs,
   the world size and the rendezvous address) into a fresh peer run directory
   and publishes one ``launch.json`` entry per peer node whose argv is
   ``python -m profiling.runners.comm._multinode <spec> <node>``.
2. Every node, the head included, spawns its ``local_size`` ranks; global rank
   ``node * local_size + local``. The ranks join one gloo group at the head's
   address and build vLLM's ``PyNcclCommunicator`` over it, so NCCL picks the
   inter-node transport itself (IB on the B200 pair), as the engine does.
3. Rank 0's payload comes back to the head through its result queue; a peer
   node's failure surfaces through its ``node<k>.exit`` / log.

Each node uses its first ``local_size`` visible GPUs. The head's share is the
L1b chunk it was given (`profiling.kernels.moe_ep_*` count only the local
ranks); a peer sees its whole node.
"""

from __future__ import annotations

import os
import pickle
import socket
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from profiling.runners.comm._launcher import MultiGpuLauncher, _raise_remote_error
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

#: Seconds a peer node has to start its ranks and the group to finish.
_GROUP_TIMEOUT_S = 1800.0


def node_count() -> int:
    """Nodes of this job (the launcher's ``VIBESIM_NODE_COUNT``), 1 outside one."""

    return int(os.environ.get("VIBESIM_NODE_COUNT", "1") or 1)


def gpus_per_node() -> int:
    """GPUs one node of this job exposes (``SLURM_GPUS_ON_NODE``, else ``/dev/nvidia<N>``)."""

    value = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if value.isdigit() and int(value) > 0:
        return int(value)
    try:
        return max(
            1,
            sum(
                1 for name in os.listdir("/dev") if name[6:].isdigit() and name.startswith("nvidia")
            ),
        )
    except OSError:
        return 1


def spans_nodes(num_gpus: int, fabric: str) -> bool:
    """True when a ``num_gpus`` group on ``fabric`` runs spread over this job's nodes.

    An InfiniBand group always spans nodes; an NVLink group spans them only when it
    is larger than one node (a GB200 NVL72 tray holds 4 GPUs, so TP16 is 4 trays
    over MNNVL).
    """

    nodes = node_count()
    if nodes < 2 or num_gpus % nodes:
        return False
    return fabric == "infiniband" or num_gpus > gpus_per_node()


def local_rank_count(num_gpus: int, fabric: str) -> int:
    """GPUs one node contributes to a ``num_gpus`` group on ``fabric``."""

    # Outside a multi-node job the spec is still well-formed; a group too large
    # for one node, or an IB group, is refused with a clear reason by the launcher.
    return num_gpus // node_count() if spans_nodes(num_gpus, fabric) else num_gpus


class MultiNodeVllmLauncher(MultiGpuLauncher):
    """A vLLM PyNCCL rank group spread evenly over the job's nodes.

    ``per_rank_fn(rank=, world_size=, communicator=, **kwargs)``, as `VllmLauncher`.
    """

    #: How ``_rank_entry`` sets the group up: ``"vllm_pynccl"``, ``"torch"`` or
    #: ``"vllm_tp"``.
    mode = "vllm_pynccl"
    backend = "gloo"

    def run(self, per_rank_fn: Callable[..., Any], **kwargs) -> Any:
        from alignment.profiler.multinode import Peers, env_overlay, topology

        topo = topology()
        if topo is None:
            raise ProfilerNotImplemented(
                "fabric='infiniband' needs a multi-node job (VIBESIM_NODE_COUNT >= 2): run "
                "kernel-profile --remote on an allocation of two or more nodes"
            )
        if self.num_gpus % topo.nodes:
            raise ProfilerNotImplemented(
                f"num_gpus={self.num_gpus} does not split evenly over {topo.nodes} nodes"
            )
        local_size = self.num_gpus // topo.nodes
        run_name = f"comm-{uuid.uuid4().hex[:12]}"
        peers = Peers(topo, run_name)
        spec_path = peers.dir / "comm_spec.pkl"
        spec = {
            "fn": per_rank_fn,
            "kwargs": kwargs,
            "world_size": self.num_gpus,
            "local_size": local_size,
            "master_addr": topo.head_addr,
            "master_port": _free_port(),
            "mode": self.mode,
            "backend": self.backend,
        }
        spec_path.write_bytes(pickle.dumps(spec))
        overlay, unset = env_overlay(dict(os.environ))
        peers.launch(
            {
                k: {
                    "argv": [sys.executable, "-m", __name__, str(spec_path), str(k)],
                    "env": overlay,
                    "unset": unset,
                    "cwd": os.getcwd(),
                    "log": str(peers.dir / f"node{k}.log"),
                    "wrapped": False,
                }
                for k in peers.ranks()
            }
        )
        try:
            return _run_node(spec, 0)
        finally:
            statuses = peers.wait_stopped(timeout=60.0)
            bad = {k: s for k, s in statuses.items() if s not in ("0",)}
            if bad:
                # The head's own result already carries the group's verdict; a
                # peer that failed after rank 0 returned is only logged.
                print(f"[multinode comm] peer exit status {bad} ({peers.dir})", file=sys.stderr)


class MultiNodeTorchLauncher(MultiNodeVllmLauncher):
    """A torch.distributed rank group spread evenly over the job's nodes.

    ``per_rank_fn(rank=, world_size=, **kwargs)`` with the process group already
    initialized on ``backend``, as `TorchMpLauncher`.
    """

    mode = "torch"

    def __init__(self, num_gpus: int, *, backend: str = "nccl"):
        super().__init__(num_gpus)
        self.backend = backend


class MultiNodeVllmTpGroupLauncher(MultiNodeVllmLauncher):
    """vLLM's own distributed state and one TP group over the job's nodes.

    ``per_rank_fn(rank=, world_size=, **kwargs)`` after
    ``init_distributed_environment`` and ``initialize_model_parallel`` with
    ``tensor_model_parallel_size=num_gpus``, as `VllmTpGroupLauncher` on one
    node (for example a TP16 group over four GB200 trays).
    """

    mode = "vllm_tp"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _run_node(spec: dict[str, Any], node_rank: int) -> Any:
    """Spawn this node's ranks; return rank 0's payload on the head."""

    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    try:
        mp.spawn(
            _rank_entry,
            args=(node_rank, spec, result_queue),
            nprocs=spec["local_size"],
            join=True,
        )
    except Exception as exc:
        if not result_queue.empty():
            ok, error_kind, payload = result_queue.get()
            if not ok:
                _raise_remote_error(error_kind, payload)
        raise KernelLaunchFailed(str(exc)) from exc
    if node_rank != 0:
        return None
    if result_queue.empty():
        raise KernelLaunchFailed("multi-node vLLM rank group returned no result")
    ok, error_kind, payload = result_queue.get()
    if not ok:
        _raise_remote_error(error_kind, payload)
    return payload


def _rank_entry(local_rank: int, node_rank: int, spec: dict[str, Any], result_queue) -> None:
    world_size = spec["world_size"]
    rank = node_rank * spec["local_size"] + local_rank
    os.environ.update(
        {
            "MASTER_ADDR": spec["master_addr"],
            "MASTER_PORT": str(spec["master_port"]),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
        }
    )
    communicator = None
    mode = spec.get("mode", "vllm_pynccl")
    if mode == "vllm_tp":
        _vllm_tp_rank_entry(rank, local_rank, spec, result_queue)
        return
    try:
        import torch
        import torch.distributed as dist

        if mode != "torch":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented("CUDA is required for a multi-node rank group")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            spec.get("backend", "gloo"),
            init_method=f"tcp://{spec['master_addr']}:{spec['master_port']}",
            rank=rank,
            world_size=world_size,
        )
        if spec.get("mode", "vllm_pynccl") == "torch":
            result = spec["fn"](rank=rank, world_size=world_size, **spec["kwargs"])
        else:
            communicator = PyNcclCommunicator(dist.group.WORLD, torch.device(f"cuda:{local_rank}"))
            if not communicator.available or communicator.disabled:
                raise ProfilerNotImplemented("vLLM PyNcclCommunicator is unavailable")
            result = spec["fn"](
                rank=rank, world_size=world_size, communicator=communicator, **spec["kwargs"]
            )
        dist.barrier()
    except Exception as exc:
        error_kind = "OOMError" if type(exc).__name__ == "OutOfMemoryError" else type(exc).__name__
        result_queue.put((False, error_kind, str(exc)))
        raise
    finally:
        if communicator is not None:
            communicator.destroy()
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    if rank == 0:
        result_queue.put((True, "", result))


def _vllm_tp_rank_entry(rank: int, local_rank: int, spec: dict[str, Any], result_queue) -> None:
    world_size = spec["world_size"]
    initialized = False
    result = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise ProfilerNotImplemented("CUDA is required for a multi-node rank group")
        torch.cuda.set_device(local_rank)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=f"tcp://{spec['master_addr']}:{spec['master_port']}",
                local_rank=local_rank,
                backend="nccl",
            )
            initialized = True
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            result = spec["fn"](rank=rank, world_size=world_size, **spec["kwargs"])
    except Exception as exc:
        error_kind = "OOMError" if type(exc).__name__ == "OutOfMemoryError" else type(exc).__name__
        result_queue.put((False, error_kind, str(exc)))
        raise
    finally:
        if initialized:
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
    if rank == 0:
        result_queue.put((True, "", result))


def main(argv: list[str]) -> int:
    spec_path, node_rank = Path(argv[0]), int(argv[1])
    # The head writes the spec before it publishes launch.json; on a shared
    # filesystem the file can trail the notice by a moment.
    deadline = time.monotonic() + 60.0
    while not spec_path.exists() and time.monotonic() < deadline:
        time.sleep(0.5)
    spec = pickle.loads(spec_path.read_bytes())
    _run_node(spec, node_rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
