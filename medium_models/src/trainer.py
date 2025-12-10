########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
if not hasattr(transformers, "is_torch_tpu_available"):
    transformers.is_torch_tpu_available = lambda: False

try:
    from transformers.utils import is_datasets_available, is_in_notebook
except ImportError:
    from transformers.file_utils import is_datasets_available, is_in_notebook

def is_torch_tpu_available(tpu_cores=None):
    return False

from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

@torch.jit.script
def _sinkhorn_loop(W: torch.Tensor, n_iter: int, epsilon: float):
    rows, cols = W.shape
    S = torch.ones(rows, 1, device=W.device, dtype=W.dtype)
    T = torch.ones(1, cols, device=W.device, dtype=W.dtype)

    for _ in range(n_iter):
        # 保持原有逻辑，但在 JIT 内部运行更快
        # unbiased=False 对应 numpy/torch 默认行为差异，这里显式计算更利于 JIT
        row_std = torch.std(W, dim=1, keepdim=True, unbiased=False).clamp(min=epsilon)
        W = W / row_std
        S = S * row_std
        
        col_std = torch.std(W, dim=0, keepdim=True, unbiased=False).clamp(min=epsilon)
        W = W / col_std
        T = T * col_std
    
    return S, T
########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    # ==============================
    # SINQ Helper Functions (NEW)
    # ==============================


    def get_sinkhorn_scales(self, param, n_iter=1, epsilon=1e-6):
        if param.dim() < 2: return None, None
        W = param.detach().float()
        rows, cols = W.shape
        if rows == 1 or cols == 1: return None, None
        
        # 调用 JIT 编译的函数
        S, T = _sinkhorn_loop(W, n_iter, epsilon)
        # === 核心修改：阻尼 (Damping) ===
        # 将 Scale 的极端值压平。
        # 解释：如果 S 是 100，开根号后变成 10。如果 S 是 0.01，变成 0.1。
        # 这样保留了方向性，但减弱了幅度差异。

        S = S.pow(0.5) 
        T = T.pow(0.5)

        # === 核心修改：截断 (Clipping) ===
        # 强制限制缩放因子的范围，防止个别参数扰动过大
        S = torch.clamp(S, min=0.1, max=10.0)
        T = torch.clamp(T, min=0.1, max=10.0)
        
        return S, T

    def apply_sinq_noise(self, z, name, param):
        update_interval = 10
        if not hasattr(self, '_sinq_cache'): self._sinq_cache = {}
        
        cache_key = name
        should_update = (self.state.global_step % update_interval == 0) or (cache_key not in self._sinq_cache)
        
        if should_update:
            S, T = self.get_sinkhorn_scales(param)
            self._sinq_cache[cache_key] = (S, T)
        else:
            S, T = self._sinq_cache[cache_key]

        if S is None or T is None: return z

        S = S.to(z.dtype)
        T = T.to(z.dtype)
        
        # 1. 原始扰动方向
        alpha = 0.5  # 这是一个超参数，建议 0.4 - 0.7
        # 这种写法既保留了方向性，又保证了 S_eff 永远不会接近 0
        S_eff = (1 - alpha) + alpha * S 
        T_eff = (1 - alpha) + alpha * T
        
        z_shaped = z * S_eff * T_eff
        
        # === 3. 行级范数对齐 (Row-wise Norm Alignment) ===
        # 这是“终极优化”的关键。
        # 我们不看整个矩阵，而是保证每一行（每个输出神经元）的扰动能量不变。
        
        # 计算原始噪声每一行的范数 (shape: [rows, 1])
        # keepdim=True 很重要，方便广播
        row_norm_orig = torch.norm(z, p=2, dim=1, keepdim=True) + 1e-6
        
        # 计算变形后噪声每一行的范数
        row_norm_shaped = torch.norm(z_shaped, p=2, dim=1, keepdim=True) + 1e-6
        
        # 计算每一行的独立缩放系数
        row_scale = row_norm_orig / row_norm_shaped
        
        # 应用缩放
        z_final = z_shaped * row_scale
        
        return z_final
    
    def apply_low_rank_sinq_noise(self, name, param, seed, scaling_factor=1.0):
        # 1. 动态计算 Sinkhorn 因子 (和之前一样，不存梯度，只用权重)
        # 注意：这里我们只计算 S (行因子) 和 T (列因子)
        # 为了极度节省显存，我们甚至不需要生成完整的 S, T 矩阵，只需要向量
        
        if param.dim() < 2:
            torch.manual_seed(seed)
            # 对 1维向量直接生成普通高斯噪声
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            return z

        rows, cols = param.shape
        if rows == 1 or cols == 1: 
            # 向量参数（如 Bias），无法做低秩分解，回退到普通高斯
            torch.manual_seed(seed)
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            return z

        # === 核心逻辑：生成低秩向量 u, v ===
        torch.manual_seed(seed)
        u = torch.normal(mean=0, std=1, size=(rows, 1), device=param.device, dtype=param.dtype)
        
        torch.manual_seed(seed + 1) # 不同的种子生成 v
        v = torch.normal(mean=0, std=1, size=(1, cols), device=param.device, dtype=param.dtype)
        
        # === 核心逻辑：应用 Sinkhorn Scaling ===
        # 我们不缓存 S, T 了，直接实时算，反正很快，省显存
        # 简化版 Sinkhorn：直接用行/列标准差近似 (Iter=1 的 Sinkhorn)
        # 这比完整循环快且不需要额外显存
        with torch.no_grad():
            W = param.detach().float()
            # 行缩放因子 (对应 u)
            row_std = torch.std(W, dim=1, keepdim=True).clamp(min=1e-6).to(param.dtype)
            # 列缩放因子 (对应 v)
            col_std = torch.std(W, dim=0, keepdim=True).clamp(min=1e-6).to(param.dtype)
            
            # 阻尼处理 (Damping)
            row_scale = row_std.pow(0.5)
            col_scale = col_std.pow(0.5)
        
        # 将 Scale 作用于低秩向量
        u_scaled = u * row_scale
        v_scaled = v * col_scale
        
        # === 核心逻辑：构造秩-1 扰动矩阵 ===
        # z = u * v^T
        # 这一步产生完整的矩阵，但它是 "结构化" 的
        z = torch.matmul(u_scaled, v_scaled)
        
        # === 范数对齐 (Norm Alignment) ===
        # 保持与全秩噪声相当的能量水平
        expected_norm = math.sqrt(rows * cols) 
        current_norm = torch.norm(z) + 1e-6
        z = z * (expected_norm / current_norm)
        
        return z

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad

    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()

    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if self.args.use_sinq:
                z = self.apply_sinq_noise(z, name, param)
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    def norm_perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z

            cname = self.retrieve_c(name)
            if cname in self.cs:
                z = z / self.cs[cname]

            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector
    
    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector

    def perturb_single_layer(self, model, layer_name, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            cname = self.retrieve_c(name)
            if cname == layer_name:
                if name in random_vector:
                    z = random_vector[name]
                else:
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                    random_vector[name] = z
                param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector

    def initialize_c(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if self.should_optim(name, param):
                self.named_parameters_to_optim.append((name, param))

        self.cs = {'embed': 0.0, 'lm_head': 0.0} 
        # OPT: embed_tokens; embed_positions
        # RoBERTa: embeddings
        self.num_params = copy.deepcopy(self.cs)
        self.num_model_layers = model.config.num_hidden_layers
        layer_name = "layers" if model.config.model_type == "opt" else "layer"
        for i in range(self.num_model_layers): 
            self.cs[f'{layer_name}.{i}.'] = 0.0
            self.num_params[f'{layer_name}.{i}.'] = 0
        
        # ZO estimation of c's
        if self.args.zo_variant != 'param_norm' and self.args.use_zo_grad_est:
            for layer in self.cs.keys():
                with torch.no_grad():
                    model, z = self.perturb_single_layer(model, layer_name=layer)
                    loss1 = self.zo_forward(model, inputs)
                    model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z, scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)

                projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)
                self.cs[layer] = torch.abs(projected_grad)

                model, z = self.perturb_single_layer(model, layer_name=layer, random_vector=z)
        
        # no need to run backprop if we are using parameter norm variant, can just measure them
        elif self.args.zo_variant == 'param_norm':
            for name, param in self.named_parameters_to_optim:
                print(name)
                ckey = self.retrieve_c(name)
                if ckey in self.cs:
                    self.cs[ckey] += torch.sum(param.data ** 2)
                    self.num_params[ckey] += param.data.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    self.cs[ckey] /= torch.sqrt(self.cs[ckey])
            
            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()
        
        # backpropagation estimation fo ZO c's
        #   this is mostly for debugging purposes to disentangle the variance from using ZO to estimate c
        #   from the effectiveness of the preconditioners
        else: 
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            for name, param in self.named_parameters_to_optim:
                if param.grad is None:
                    print(name)
                else:
                    ckey = self.retrieve_c(name)
                    if ckey in self.cs:
                        self.cs[ckey] += torch.sum(param.grad ** 2)
                        self.num_params[ckey] += param.grad.numel()

            # take sqrt to get norm
            for ckey in self.cs:
                self.cs[ckey] = torch.sqrt(self.cs[ckey])
                if self.args.scale_norm_by_num_params:
                    self.cs[ckey] /= torch.sqrt(self.num_params[ckey])

            for ckey in self.cs:
                if self.cs[ckey] != 0:
                    self.cs[ckey] = self.cs[ckey].detach().item()

        self.layer_names = list(self.cs.keys())
        model.zero_grad()

    def retrieve_c(self, param_name):
        for c_name in self.cs.keys():
            if c_name in param_name:
                return c_name

        return '' # these parameters are likely not being used in the forward pass

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError
        # print("Sample %d zs" % (noise_sample_time))

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        import torch.distributed as dist
        if self.args.local_rank != -1 and dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            world_size = 1
            if self.args.local_rank != -1 and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * world_size
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if hasattr(self, '_sinq_cache'):
                self._sinq_cache = {}

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                # estimate c's (param or grad norm) on epoch 0
                if epoch == 0 and step == 0 and self.args.zo_variant is not None:
                    self.initialize_c(model, inputs)
                elif step == 0 and self.args.zo_variant is not None and self.args.recompute_norms:
                    self.initialize_c(model, inputs)
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                    
                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))

                    if self.args.zo_by_layer:
                        assert not self.args.efficient_zero_order, 'did not implement preconditioned ZO for efficient ZO yet'
                        assert self.args.zero_order_use_trainer_optim, 'preconditioned ZO requires using the trainer optimizer'
                        num_zs = self.get_num_samples()
                        layers = [np.random.choice(self.layer_names)] if self.args.pc_rnd_layer else self.layer_names

                        # for each layer: perturb only that layer and store the gradient estimates in the grad buffer
                        for layer in self.layer_names:
                            for _ in range(num_zs):
                                c_i = self.cs[layer]
                                with torch.no_grad():
                                    c_i = 1.0 if c_i == 0 else c_i # if the scaling is 0, just reset it to 1 so that there can eventually be some gradient to those layers 
                                    model, random_vector = self.perturb_single_layer(model, layer, scaling_factor=1.0/c_i)
                                    loss1 = self.zo_forward(model, inputs)
                                    model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=-2.0/c_i)
                                    loss2 = self.zo_forward(model, inputs)
                                    model, random_vector = self.perturb_single_layer(model, layer, random_vector=random_vector, scaling_factor=1.0/c_i)

                                projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)
                                # scale grad according to number of zs sampled
                                if not self.args.scale_lr_with_samples:
                                    projected_grad = projected_grad / float(num_zs)
                                
                                for name, param in self.named_parameters_to_optim:
                                    if self.retrieve_c(name) == layer:
                                        z_tilde = random_vector[name] * c_i

                                        if param.grad is None:
                                            param.grad = projected_grad * z_tilde
                                        else:
                                            param.grad += projected_grad * z_tilde

                                # note that  | E_z [ <z, grad of one layer > ] | is equal to norm of grad for that layer for gaussian z
                                # leverages this fact to update the grad norms
                                if self.args.zo_variant == 'grad_norm' and self.args.norm_running_update:
                                    self.cs[layer] = torch.abs(projected_grad)
                    else:
                        # get number of zs to sample
                        num_zs = self.get_num_samples()
                        if num_zs > 1:
                            assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

                        for _ in range(num_zs):
                            # prepare for sampling new zs
                            random_vector = None
                            if self.args.efficient_zero_order:
                                random_seed = np.random.randint(1000000000)

                            with torch.no_grad():
                                # first function evaluation
                                if self.args.efficient_zero_order:
                                    model = self.efficient_perturb_parameters(model, random_seed)
                                elif self.args.zo_variant is not None:
                                    model, random_vector = self.norm_perturb_parameters(model)
                                else:
                                    model, random_vector = self.perturb_parameters(model)
                                loss1 = self.zo_forward(model, inputs)

                                # second function evaluation
                                if self.args.efficient_zero_order:
                                    model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                                elif self.args.zo_variant is not None:
                                    model, random_vector = self.norm_perturb_parameters(model, random_vector, scaling_factor=-2)
                                else:
                                    model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)                 
                                loss2 = self.zo_forward(model, inputs)

                            projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)

                            # scale grad according to accumulation
                            if self.args.gradient_accumulation_steps > 1:
                                assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                                projected_grad = projected_grad / self.args.gradient_accumulation_steps
                            
                            # scale grad according to number of zs sampled
                            if not self.args.scale_lr_with_samples:
                                projected_grad = projected_grad / float(num_zs)

                            # store gradient in parameter buffer if using trainer
                            # o/w, the loop will exit after one round and the update will be applied directly (see below)
                            if self.args.zero_order_use_trainer_optim:
                                if self.args.efficient_zero_order:
                                    # print(random_seed)
                                    torch.manual_seed(random_seed)
                                
                                for name, param in self.named_parameters_to_optim:
                                    # recover noise used in perturbations
                                    if self.args.efficient_zero_order:
                                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                    else:
                                        z = random_vector[name]

                                    if self.args.zo_variant is not None and not self.args.change_grad_estimate:
                                        cname = self.retrieve_c(name)
                                        if cname in self.cs:
                                            z = z * self.cs[cname]

                                    if param.grad is None:
                                        param.grad = projected_grad * z
                                    else:
                                        param.grad += projected_grad * z

                            # reset model back to its parameters at start of step
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed)
                            elif self.args.zo_variant is not None:
                                model, random_vector = self.norm_perturb_parameters(model, random_vector)   
                            else:
                                model, random_vector = self.perturb_parameters(model, random_vector)

                    # apply gradient updates
                    # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
                    if self.args.zero_order_use_trainer_optim:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                        ):
                            # Gradient norm clipping
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            # Update the parameters and step scheduler
                            optimizer.step()
                            scheduler.step()
                        
                            # logging
                            if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                if not self.args.zero_order_clip_grad:
                                    norm = 0.0
                                    for _, p in model.named_parameters():
                                        if p.grad is not None:
                                            norm += torch.sum(p.grad ** 2)
                                    norm = torch.sqrt(norm)
                                logs["grad_norm"] = norm.item()
                                logs["learning_rate"] = (
                                    scheduler.get_last_lr()[0]
                                    if version.parse(torch.__version__) >= version.parse("1.4")
                                    else scheduler.get_lr()[0]
                                )
                                logs["num_zs"] = num_zs
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                self.log(logs)
                                logger.info(str(logs))
                            
                            model.zero_grad()
                            self.state.global_step += 1
                            self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    # if not using the trainer, the updates are resampled and directly applied to the parameters
                    else:
                        # Efficient mode 
                        # WARNING: no gradient accumulation when not storing the grad
                        assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
                        assert self.args.zero_order_sample_scheduler is None
                        assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

                        if self.args.efficient_zero_order:
                            torch.manual_seed(random_seed)     
                        for name, param in self.named_parameters_to_optim:
                            if self.args.efficient_zero_order:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            else:
                                z = random_vector[name]
                             # 保持与 perturb 阶段一致
                            if self.args.use_sinq:
                                z = self.apply_sinq_noise(z, name, param)
                            param.data = param.data - self.args.learning_rate * (projected_grad * z + self.args.weight_decay * param.data)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                logs["learning_rate"] = self.args.learning_rate
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                self.log(logs)
                                logger.info(str(logs))


                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    
                    # Debug information
                    # print("%.5f, %.5f" % (loss1.item(), loss2.item()))
                    # print("Loss: %.10f, projected_grad: %.5f" % (loss1, projected_grad))

                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            self.log(logs)
                            logger.info(str(logs))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
