''' Simplified Proximal Policy Optimization (MPPO) algorithm. 
Currently support Like REINFORCED, Re-Max, which three models exist simultaneously.

Date: 2024-02-02 14:51:48
LastEditors: Dylancer1998 bodcoder@gmail.com
LastEditTime: 2024-02-02 14:51:52
'''

import torch
from megatron.core import mpu
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core.enums import ModelType
from megatron.training import get_model, get_optimizer_param_scheduler
from megatron.training import setup_model_and_optimizer
from megatron.utils import unwrap_model
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.optimizer import get_megatron_optimizer
from megatron_rlhf.generation.api import generate_and_post_process # patched based on megatron.text_generation.api


def _build_train_valid_dataloaders(
    train_dataset, 
    valid_dataset, 
    task_collate_fn=None
):
    """
    Build train and validation dataloaders.
    """
    args = get_args()
    
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    # Train dataloader.
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=task_collate_fn
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank)
    # Validation dataloader.
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=task_collate_fn
    )
    return train_dataloader, valid_dataloader


def load_reference_model(model_provider):
    # setup model
    model = get_model(model_provider, 
                      model_type=ModelType.encoder_or_decoder,
                      wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    torch.distributed.barrier()
    
    if isinstance(model, list):
        model = model[0]
    return model


def finetune(
    datasets_provider,
    model_provider,
    forward_step,
    task_collate_fn=None
):
    """
    Main function for simplified-PPO
    """
    args = get_args()
    timers = get_timers()
    
    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder', log_level=0).start()
    if args.epochs > 0:
        train_dataset, valid_dataset = datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn)
    else:
        args.train_iters = 0
    timers('train/valid/test dataset/dataloder').stop()
    
    print(f"dataset len: {len(train_dataset)} | {len(valid_dataset)}")
    
    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer', log_level=0).start()
    model_provider_actor, model_provider_refer, _ = model_provider()
    actor_model, actor_optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider_actor, model_type=ModelType.encoder_or_decoder)
    timers('model and optimizer').stop()
    
    # Load reference model
    timers('refer-model-setup', log_level=0).start(barrier=True)
    reference_model = load_reference_model(model_provider_refer)
    reference_model.recompute_granularity = None
    timers('refer-model-setup').stop()
    
    for epoch in range(args.epochs):
        # make experience
        
        import torch.distributed as dist
        # start to rollouts
        for batch_prompts in train_dataloader:
            print(f"{dist.get_rank()} batch_prompts: {batch_prompts}")
            rets = generate_and_post_process(
                    reference_model,
                    prompts=batch_prompts,
                    tokens_to_generate=500, # api to open
                    temperature=0.8, # api to openi
                )
            if mpu.is_pipeline_first_stage():
                sequences, _, _, _ = rets
                print(f"{dist.get_rank()} sequences: {sequences}")

            break
        
    
    