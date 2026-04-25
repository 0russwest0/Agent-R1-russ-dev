"""Agent-R1 wrappers for verl new-engine workers."""

import sys

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name
from verl.utils.seqlen_balancing import roundup_divisible
from verl.workers.engine import utils as engine_utils
from verl.workers.engine_workers import (
    ActorRolloutRefWorker as VerlActorRolloutRefWorker,
)
from verl.workers.engine_workers import (
    TrainingWorker as VerlTrainingWorker,
)

_ORIGINAL_PREPARE_MICRO_BATCHES = engine_utils.prepare_micro_batches
_PATCHED = False


def _split_fixed_micro_batches(
    data: TensorDict,
    micro_batch_size_per_gpu: int,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
) -> list[TensorDict]:
    num_micro_batches = (len(data) + micro_batch_size_per_gpu - 1) // micro_batch_size_per_gpu
    if min_num_micro_batch is not None:
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches_tensor = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches_tensor, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches_tensor.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    assert num_micro_batches <= len(data), (
        f"cannot split {len(data)} samples into {num_micro_batches} non-empty micro batches without padding"
    )

    micro_bsz_idx = []
    remaining = len(data)
    offset = 0
    for remaining_batches in range(num_micro_batches, 0, -1):
        split_size = (remaining + remaining_batches - 1) // remaining_batches
        assert split_size <= micro_batch_size_per_gpu, (
            f"micro batch size {split_size} exceeds configured {micro_batch_size_per_gpu}"
        )
        micro_bsz_idx.append(list(range(offset, offset + split_size)))
        offset += split_size
        remaining -= split_size

    return [tu.index_select_tensor_dict(data, partition) for partition in micro_bsz_idx]


def _prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    if use_dynamic_bsz:
        return _ORIGINAL_PREPARE_MICRO_BATCHES(
            data=data,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
        )

    micro_batch_size_per_gpu = tu.get_non_tensor_data(data=data, key="micro_batch_size_per_gpu", default=None)
    assert micro_batch_size_per_gpu is not None, "micro_batch_size_per_gpu must be set when use_dynamic_bsz is False"
    micro_batches = _split_fixed_micro_batches(
        data=data,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        dp_group=dp_group,
        num_batches_divided_by=num_batches_divided_by,
        same_micro_num_in_dp=same_micro_num_in_dp,
        min_num_micro_batch=min_num_micro_batch,
    )
    return micro_batches, None


def _install_agent_r1_micro_batching_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    engine_utils.prepare_micro_batches = _prepare_micro_batches

    for module_name in (
        "verl.workers.engine.fsdp.transformer_impl",
        "verl.workers.engine.megatron.transformer_impl",
        "verl.workers.engine.veomni.transformer_impl",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.prepare_micro_batches = _prepare_micro_batches

    _PATCHED = True


class TrainingWorker(VerlTrainingWorker):
    def __init__(self, *args, **kwargs):
        _install_agent_r1_micro_batching_patch()
        super().__init__(*args, **kwargs)


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        _install_agent_r1_micro_batching_patch()
        super().__init__(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        _install_agent_r1_micro_batching_patch()
        return super().init_model()
