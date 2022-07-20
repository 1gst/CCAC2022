import copy
from typing import Dict, Union, Any, Optional, Callable, List, Tuple

import torch
from apex import amp
from datasets import Dataset
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback

# 重载Trainer
# from transformers.trainer_pt_utils import smp_forward_backward
from transformers.utils import is_sagemaker_mp_enabled

from utils.adversarial import FGM
from utils.focal_loss import FocalLoss


class MyTrainer(Trainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 use_fgm=False):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.use_fgm = use_fgm

    # 重写training_step方法
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        inputs_adv = copy.deepcopy(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        #是否使用对抗网络
        if self.use_fgm:
            # 定义FGM（对抗网络）
            fgm = FGM(model, epsilon=1, emb_name='word_embeddings')
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            loss_adv = self.compute_loss(model, inputs_adv)
            loss_adv.backward()
            fgm.restore()
        return loss.detach()

    # 重写compute_loss方法，该方法主要定义了 forward和loss的计算过程
    def compute_loss(self, model, inputs, return_outputs=False):
        # 即使处理数据的时候只有label标签，在trainer内部依旧会处理成labels标签
        # 见https://zhuanlan.zhihu.com/p/416002644
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # 先声明损失函数
        loss_fct = FocalLoss(alpha=torch.FloatTensor([0.75,0.25]))
        # loss_fct = CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, {'outputs': logits}) if return_outputs else loss
