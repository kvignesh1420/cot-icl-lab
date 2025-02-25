"""Training"""

import os
import argparse
from datetime import timedelta

# set logging level to INFO
import logging

logging.basicConfig(level=logging.INFO)


import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import torch.distributed as dist
from torch.distributed.elastic.utils.data import ElasticDistributedSampler

from torch.utils.data import DataLoader
from transformers import GenerationConfig

from tokenized_cot_icl.core.args import Args, IGNORE_INDEX
from tokenized_cot_icl.core.task_card import TASK_CARD
from tokenized_cot_icl.core.data import (
    TokenizedDataset,
    EvalTokenizedDataset,
)
from tokenized_cot_icl.core.models import MODEL_REGISTRY
from tokenized_cot_icl.core.utils import (
    set_random_seed,
    get_mlflow_client,
    prepare_run_name,
)


class Trainer:
    def __init__(self, args: Args, device_id: int, mlflow_run_id: str):
        self.run_name = prepare_run_name(args=args)
        self.args = args
        self.device_id = device_id
        self.mlflow_run_id = mlflow_run_id
        self.train_dataset_clz = TokenizedDataset
        self.eval_dataset_clz = EvalTokenizedDataset
        self.collate_fn = None
        self.create_model()
        self.create_dataloaders()

    def create_model(self):
        set_random_seed(self.args.seed)
        model_type = self.args.model_type
        assert model_type in MODEL_REGISTRY, f"Model type {model_type} not supported."
        self.model = MODEL_REGISTRY[model_type](self.args)
        self.model.cuda(self.device_id)
        self.model = DistributedDataParallel(self.model, device_ids=[self.device_id])
        # create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        if self.device_id == 0:
            self.mlflow_client = get_mlflow_client()
            logging.info(f"Rank: {self.device_id}. MLFlow run_id: {self.mlflow_run_id}")
            for k, v in self.args.__dict__.items():
                self.mlflow_client.log_param(run_id=self.mlflow_run_id, key=k, value=v)

    def create_dataloaders(self):
        # train
        set_random_seed(self.args.seed)
        train_dataset = self.train_dataset_clz(self.args)
        train_sampler = ElasticDistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        # eval (set a different random seed for eval dataset since it is materialized during creation)
        set_random_seed(self.args.seed + 1000)
        eval_dataset = self.eval_dataset_clz(self.args)
        assert torch.allclose(
            train_dataset.embeddings.weight, eval_dataset.embeddings.weight
        ), "Embeddings are not shared between train and eval datasets."
        assert len(train_dataset) == self.args.n_tasks, (
            f"Train dataset length {len(train_dataset)} does not match number of tasks {self.args.n_tasks}."
        )
        assert len(eval_dataset) == self.args.n_eval_tasks, (
            f"Eval dataset length {len(eval_dataset)} does not match number of tasks {self.args.n_eval_tasks}."
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def metrics_at_init(self):
        logging.info("Calculating eval metrics at init and saving the checkpoint.")
        # Evaluate at init
        step = 0
        self.evaluate(step=step)
        self.cot_evaluate(step=step)
        # Save checkpoint at init
        self.save_checkpoint(step=step)

    def train(self):
        # print model summary and parameter count
        logging.info(self.model)
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

        # get metrics with the initialized model as a baseline
        self.metrics_at_init()
        # training loop
        self.model.train()
        # reset random seed since data from the loader is generated on the fly
        set_random_seed(self.args.seed + self.device_id * 10)
        for epoch in range(self.args.num_epochs):
            self.train_loader.batch_sampler.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader):
                training_step = epoch * len(self.train_loader) + batch_idx + 1
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].cuda(self.device_id)
                attention_mask = batch["attention_mask"].cuda(self.device_id)
                labels = batch["labels"].cuda(self.device_id)

                # Forward pass
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Log to MLFlow
                if (
                    self.device_id == 0
                    and training_step % self.args.log_every_n_steps == 0
                ):
                    self.mlflow_client.log_metric(
                        run_id=self.mlflow_run_id,
                        key="train_loss",
                        value=loss.item(),
                        step=training_step,
                    )

                # Evaluate
                if training_step % self.args.eval_every_n_steps == 0:
                    self.evaluate(step=training_step)
                    self.cot_evaluate(step=training_step)
                    self.model.train()

                # Save checkpoint
                if training_step % self.args.checkpoint_every_n_steps == 0:
                    self.save_checkpoint(step=training_step)

        if self.device_id == 0:
            self.mlflow_client.set_terminated(self.mlflow_run_id)

    @torch.no_grad()
    def evaluate(self, step: int) -> None:
        self.model.eval()
        total_loss = 0.0
        for batch in self.eval_loader:
            input_ids = batch["input_ids"].cuda(self.device_id)
            attention_mask = batch["attention_mask"].cuda(self.device_id)
            labels = batch["labels"].cuda(self.device_id)

            # Forward pass
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_loader)
        if self.device_id == 0:
            self.mlflow_client.log_metric(
                run_id=self.mlflow_run_id,
                key="eval_loss",
                value=avg_loss,
                step=step,
            )

    @torch.no_grad()
    def cot_evaluate(self, step: int) -> None:
        # Evaluate the model with CoT and without teacher forcing on the last example.
        self.model.eval()
        max_new_tokens = self.args.chain_length if self.args.enable_cot else 1
        # Initialize chain losses and accuracies
        chain_losses = [0.0 for _ in range(max_new_tokens)]
        chain_prediction_acc = [
            {"correct": 0.0, "total": 0.0} for _ in range(max_new_tokens)
        ]
        for batch in self.eval_loader:
            input_ids = batch["cot_eval"]["input_ids"].cuda(self.device_id)
            attention_mask = batch["cot_eval"]["attention_mask"].cuda(self.device_id)
            last_example_cot = batch["cot_eval"]["last_example_cot"].cuda(
                self.device_id
            )

            # Generate the chain of tokens using model forward passes without teacher forcing
            # i.e we use the .generate() method.

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                # sampler
                do_sample=False,
                # distribution, we do not set temperature since do_sample=False
                # temperature=self.args.temperature,
                # token ids
                pad_token_id=self.args.pad_token_id,
                eos_token_id=self.args.eos_token_id,
                bos_token_id=self.args.bos_token_id,
                use_cache=True,
                output_logits=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # since we are using DistributedDataParallel, we need to use model.module
            output = self.model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            assert len(output.logits) == max_new_tokens

            # compute the chain prediction accuracies and losses
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            for chain_idx, logits in enumerate(output.logits):
                logits = logits.view(-1, self.args.vocab_size)
                labels = last_example_cot[:, chain_idx].view(-1)
                assert logits.shape[0] == labels.shape[0]
                # accumulate loss
                loss = loss_fct(logits, labels)
                chain_losses[chain_idx] += loss.item()
                # accumulate correct predictions
                predictions = torch.argmax(logits, dim=-1)
                correct = torch.eq(predictions, labels).sum().item()
                chain_prediction_acc[chain_idx]["correct"] += correct
                chain_prediction_acc[chain_idx]["total"] += labels.shape[0]

        if self.device_id == 0:
            for chain_idx, (loss, pred_info) in enumerate(
                zip(chain_losses, chain_prediction_acc)
            ):
                avg_loss = loss / len(self.eval_loader)
                self.mlflow_client.log_metric(
                    run_id=self.mlflow_run_id,
                    key=f"cot_eval_loss_chain_idx_{chain_idx}",
                    value=avg_loss,
                    step=step,
                )
                accuracy = pred_info["correct"] / pred_info["total"]
                self.mlflow_client.log_metric(
                    run_id=self.mlflow_run_id,
                    key=f"cot_eval_accuracy_chain_idx_{chain_idx}",
                    value=accuracy,
                    step=step,
                )

    def save_model(self):
        if self.device_id == 0:
            model_dir = os.path.join(self.args.output_dir, self.run_name, "final_model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            logging.info(f"Saving model to {model_dir}")
            self.model.module.save_pretrained(model_dir)

    def save_checkpoint(self, step: int):
        if self.device_id == 0:
            checkpoint_dir = os.path.join(
                self.args.output_dir, self.run_name, "checkpoints", str(step)
            )
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            logging.info(f"Saving checkpoint to {checkpoint_dir}")
            self.model.module.save_pretrained(checkpoint_dir)

    def save_eval_dataset(self):
        if self.device_id == 0:
            eval_dataset_dir = os.path.join(
                self.args.output_dir, self.run_name, "eval_dataset"
            )
            if not os.path.exists(eval_dataset_dir):
                os.makedirs(eval_dataset_dir)
            logging.info(f"Saving eval dataset to {eval_dataset_dir}")
            torch.save(
                self.eval_loader.dataset.data,
                os.path.join(eval_dataset_dir, "eval_data.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_card_key", type=int, required=True)
    parser.add_argument("--mlflow_run_id", type=str, required=True)
    parser_args = parser.parse_args()

    task_card_key = parser_args.task_card_key
    assert task_card_key in TASK_CARD, f"Invalid task_card_key: {task_card_key}"

    args = TASK_CARD[task_card_key]

    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    logging.info(f"=> set cuda device = {device_id}")

    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=300)
    )

    trainer = Trainer(
        args=args, device_id=device_id, mlflow_run_id=parser_args.mlflow_run_id
    )
    trainer.train()
    trainer.save_model()
    trainer.save_eval_dataset()
