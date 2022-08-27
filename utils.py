"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

from xmlrpc.client import Boolean
import torch.distributed as dist
import torch
import os
from config import ConfigType


def init_for_distributed(rank: int, opts: ConfigType) -> int:

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl',
                            init_method="env://",
                            world_size=len(opts.gpu_ids),
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    dist.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return local_gpu_id


def setup_for_distributed(is_master: bool):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
