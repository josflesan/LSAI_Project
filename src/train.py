import os
import time
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer
from torch.distributed._tensor import Shard, Replicate
from dataset.dataset import CollatorForCLM, ParquetDataset
from model.model import Transformer, TransformerModelArgs
from utils.utils import (
    build_lr_scheduler,
    clip_grad_norm_,
    get_args,
    get_num_params,
    get_num_flop_per_token,
    init_logger,
    logger,
    PRECISION_STR_TO_DTYPE,
    set_default_dtype,
    verify_min_gpu_count,
)
from utils.distributed_utils import init_distributed, compare_tensors
from pathlib import Path

# -------- GPU check --------

_min_gpu_count = 1

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(
        f"Unable to locate sufficienct {_min_gpu_count} GPUs to run this example. Exiting."
    )
    sys.exit()

# ---------------------------

# Distributed parameters
RANK, LOCAL_RANK, WORLD_SIZE = (0, 0, 1)
random.seed(32)
torch.manual_seed(32)  # TODO: remove if running new experiment

# Define layer TP plans
# 1. Feedforward Transformer Block
# 2. Feedforward + Self-Attention Block
# 3. Transformers + Embeddings + Output Linear
feedforward_tp_plan = {
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}

attention_tp_plan = {
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
}

global_plan = {
    "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
    "output": ColwiseParallel(output_layouts=Replicate()),
}


def train(args, tp_mesh=None, dp_mesh=None):
    logger.info(f"Experiment args: {args}")

    # Init
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

    # Set up DataLoader
    logger.info("Setting up DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(
        args.dataset,
        tokenizer,
        args.sequence_length,
        args.batch_size * args.training_steps,
    )
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=train_collator
    )
    train_dl_iterator = iter(train_dl)

    # Set up Model
    logger.info("Setting up Model...")
    model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32768,
        seq_len=args.sequence_length,
    )
    with set_default_dtype(model_dtype):
        model = Transformer(model_config).to("cuda")

    if args.compile:
        logger.info("Using `torch.compile`")
        model = torch.compile(model, fullgraph=True)

    # model.to(device)

    # Parallelize each feedforward layer in the transformer blocks
    if args.tensor_parallel:
        for layer_id, transformer_block in model.layers.items():
            # Adjust attention module to use local number of heads
            if (
                args.tp_parallel_type == "attention"
                or args.tp_parallel_type == "global"
            ):
                attn_layer = transformer_block.attention
                attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
                attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=feedforward_tp_plan
                if args.tp_parallel_type == "feedforward"
                else {**feedforward_tp_plan, **attention_tp_plan},
            )

        if args.tp_parallel_type == "global":
            model = parallelize_module(model, tp_mesh, global_plan)

        if args.data_parallel:
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

    model.train()

    # Build Optimizers & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer
    )
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

    # Utils
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
    )

    ntokens_since_last_log = 0
    ntraining_tokens_since_last_log = 0
    time_last_log = time.perf_counter()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    max_peak_memory = 0

    if RANK == 0:
        logger.info("Starting training!")
    train_step = 0
    while train_step < args.training_steps:
        train_step += 1

        # Profiling
        if args.profile and args.profile_step_start == train_step:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        input_ids, labels = next(train_dl_iterator)
        local_batch_size = input_ids.size(0)
        ntokens_since_last_log += local_batch_size * args.sequence_length
        num_items_in_batch = labels.ne(-100).sum()
        ntraining_tokens_since_last_log += num_items_in_batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        # Compute loss with properly shaped tensors
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
        )
        loss = (
            loss / num_items_in_batch
        )  # Use actual number of elements in case we truncated
        del logits
        loss.backward()

        # Track peak memory
        peak_memory = torch.cuda.max_memory_allocated(device)  # in bytes
        max_peak_memory = max(max_peak_memory, peak_memory)

        # Log loss to tensorboard
        if RANK == 0:
            writer.add_scalar("Loss/train (CE)", loss, train_step)
            writer.add_scalar(
                "Memory/Peak Memory (GB)", peak_memory / (1024**3), train_step
            )

        # Clip gradients
        clip_grad_norm_(model.parameters(), args.grad_max_norm, tp_mesh)

        optimizer.step()
        lr_scheduler.step()

        # Logging
        if train_step == 1 or train_step % args.logging_frequency == 0:
            time_delta = time.perf_counter() - time_last_log
            # tokens per second per device, abbreviated as tps
            tps = ntokens_since_last_log / time_delta
            mfu = 100 * num_flop_per_token * tps / 989e12
            tflops = num_flop_per_token * tps / 1e12
            training_tps = ntraining_tokens_since_last_log / time_delta

            # Only print on one rank
            # if RANK == 0:
            logger.info(
                f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100 * training_tps / tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}"
            )

            ntokens_since_last_log = 0
            ntraining_tokens_since_last_log = 0
            time_last_log = time.perf_counter()

        # Profiling
        if args.profile and args.profile_step_end == train_step:
            torch.cuda.cudart().cudaProfilerStop()

    all_peak_memory = [torch.cuda.max_memory_allocated()]
    all_peak_memory = torch.tensor(all_peak_memory, device=device)
    dist.all_reduce(
        all_peak_memory, op=dist.ReduceOp.MAX
    )  # All reduce max to get the peak memory usage across ranks
    if RANK == 0:
        logger.info(
            f"Global max peak memory across all ranks: {all_peak_memory.item() / (1024**3):.2f} GB"
        )

    if RANK == 0:
        logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    args = get_args()

    # Define device mesh
    tp_mesh = None
    dp_mesh = None
    if args.tensor_parallel or args.data_parallel:
        RANK, LOCAL_RANK, WORLD_SIZE = init_distributed()

    if args.tensor_parallel and args.data_parallel:
        device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        dp_mesh = device_mesh["dp"]
        tp_mesh = device_mesh["tp"]
    elif args.tensor_parallel:
        device_mesh = init_device_mesh("cuda", (4,))
        tp_mesh = device_mesh
    elif args.data_parallel:
        device_mesh = init_device_mesh("cuda", (4,))
        dp_mesh = device_mesh

    # Create experiment directory if it doesn't exist
    output_dir = Path(
        f"/iopsstor/scratch/cscs/{args.user}/LSAI_Project/logs/tensorboard/{args.experiment}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir)
    train(args, tp_mesh, dp_mesh)
    writer.close()  # Close the Tensorboard writer
