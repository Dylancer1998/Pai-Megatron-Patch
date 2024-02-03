'''
Date: 2024-02-02 14:51:20
LastEditors: Dylancer1998 bodcoder@gmail.com
LastEditTime: 2024-02-02 17:10:17
'''

from functools import partial
import torch

from megatron import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.initialize import initialize_megatron
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_ltor_masks_and_position_ids
from megatron_rlhf.data import PromptDataset, zero_pad_sequences
from megatron_rlhf.train.sppo import finetune
from megatron_patch.model.llama2.gpt_model import GPTModel
from megatron_patch.tokenizer import get_tokenizer, build_tokenizer
from megatron_patch.arguments import get_patch_args
from megatron.arguments import core_transformer_config_from_args


def model_provider():
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    def model_provider_actor(pre_process=True, post_process=True):
        return GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    
    def model_provider_refer(pre_process=True, post_process=True):
        # if parallel_output is not True, gather the output to the first rank
        parallel_output = True
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            parallel_output = False
        
        # for inference
        config.recompute_granularity = None
        
        return GPTModel(
            config=config,
            num_tokentypes=0,
            parallel_output=parallel_output,
            pre_process=pre_process,
            post_process=post_process
        )
    return model_provider_actor, model_provider_refer, None


def train_valid_datasets_provider():
    args = get_args()
    build_tokenizer(args)
    train_dataset = PromptDataset(path=args.train_data_path)
    valid_dataset = PromptDataset(path=args.valid_data_path)
    return train_dataset, valid_dataset


def forward_step(data_iterator, model):
    args = get_args()
    tokenizer = get_tokenizer()

    try:
        data_iterator = next(data_iterator)
    except BaseException:
        data_iterator = data_iterator

    tokens = data_iterator['input_ids'].long().cuda().contiguous()
    labels = data_iterator['labels'].long().cuda().contiguous()

    tokens = tokens[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.pad_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)

    logits = model(input_ids=tokens,
                   position_ids=position_ids,
                   attention_mask=attention_mask)

    def loss_func(loss_mask, logits):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), labels.contiguous())
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'lm loss': averaged_loss[0]}

    return logits, partial(loss_func, loss_mask)


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_patch_args)

    finetune(datasets_provider=train_valid_datasets_provider,
             model_provider=model_provider,
             forward_step=forward_step)
