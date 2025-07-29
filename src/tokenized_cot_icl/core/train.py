"""Training"""

import argparse
import json

# set logging level to INFO
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from tokenized_cot_icl.core.args import IGNORE_INDEX, Args
from tokenized_cot_icl.core.data.data import EvalTokenizedDataset, TokenizedDataset, special_token_collate_fn
from tokenized_cot_icl.core.models import MODEL_REGISTRY
from tokenized_cot_icl.core.task_card import TASK_CARD
from tokenized_cot_icl.core.utils import create_metric_logger, prepare_run_name, set_random_seed

logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, args: Args, device_id: int, world_size: int):
        self.run_name = prepare_run_name(args)
        self.args = args
        self.device_id = device_id
        self.world_size = world_size
        self.train_dataset_clz = TokenizedDataset
        self.eval_dataset_clz = EvalTokenizedDataset
        self.collate_fn = None
        self.create_output_dir()
        self.create_model()
        self.create_dataloaders()
        self.metric_logger = create_metric_logger()

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

    def create_dataloaders(self):
        # train
        # The seed is set within the dataset class itself.
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
        # self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        # eval (set a different random seed for eval dataset since it is materialized during creation)
        # The seed is set within the dataset class itself.
        eval_dataset = self.eval_dataset_clz(self.args)
        assert torch.allclose(train_dataset.embeddings.weight, eval_dataset.embeddings.weight), (
            "Embeddings are not shared between train and eval datasets."
        )
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

    def gather_stats_at_rank_0(self, rank: int, stats_tensor: torch.Tensor):
        if rank == 0:
            gathered_tensors = [torch.zeros_like(stats_tensor) for _ in range(self.world_size)]
        else:
            gathered_tensors = None

        dist.gather(stats_tensor, gather_list=gathered_tensors if rank == 0 else None, dst=0)

        if rank == 0:
            total_sum = torch.stack(gathered_tensors).sum()
            return total_sum
        return None

    def train(self):
        # print model summary and parameter count
        logging.info(self.model)
        logging.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

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
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # all gather the cot/standard example counts across ranks
                num_cot_examples_batch = batch["num_cot_examples"].cuda(self.device_id)
                num_standard_examples_batch = batch["num_standard_examples"].cuda(self.device_id)
                # all gather the counts across ranks
                all_ranks_num_cot_examples = self.gather_stats_at_rank_0(
                    rank=self.device_id, stats_tensor=num_cot_examples_batch
                )
                all_ranks_num_standard_examples = self.gather_stats_at_rank_0(
                    rank=self.device_id, stats_tensor=num_standard_examples_batch
                )

                # Log loss and example stats
                if training_step % self.args.log_every_n_steps == 0:
                    self.metric_logger.log_metrics(metrics={"train_loss": loss.item()}, step=training_step)

                    self.metric_logger.log_metrics(
                        metrics={"num_cot_examples": all_ranks_num_cot_examples.item()}, step=training_step
                    )
                    self.metric_logger.log_metrics(
                        metrics={"num_standard_examples": all_ranks_num_standard_examples.item()}, step=training_step
                    )

                # Evaluate
                if training_step % self.args.eval_every_n_steps == 0:
                    self.evaluate(step=training_step)
                    self.cot_evaluate(step=training_step)
                    self.model.train()

                # Save checkpoint
                if training_step % self.args.checkpoint_every_n_steps == 0:
                    self.save_checkpoint(step=training_step)

        self.metric_logger.close()

    @torch.no_grad()
    def evaluate(self, step: int) -> None:
        self.model.eval()
        total_loss = 0.0
        for batch in self.eval_loader:
            input_ids = batch["input_ids"].cuda(self.device_id)
            attention_mask = batch["attention_mask"].cuda(self.device_id)
            labels = batch["labels"].cuda(self.device_id)

            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_loader)
        self.metric_logger.log_metrics(
            {"eval_loss": avg_loss},
            step=step,
        )

    @torch.no_grad()
    def cot_evaluate(self, step: int) -> None:
        # Evaluate the model with CoT and without teacher forcing on the last example.
        self.model.eval()
        max_new_tokens = self.args.chain_length_choices[-1] if self.args.prompt_strategy == "cot" else 1
        # Initialize chain losses and accuracies
        chain_losses = [0.0 for _ in range(max_new_tokens)]
        chain_prediction_acc = [{"correct": 0.0, "total": 0.0} for _ in range(max_new_tokens)]
        for batch in self.eval_loader:
            input_ids = batch["cot_eval"]["input_ids"].cuda(self.device_id)
            attention_mask = batch["cot_eval"]["attention_mask"].cuda(self.device_id)
            last_example_cot = batch["cot_eval"]["last_example_cot"].cuda(self.device_id)

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

        for chain_idx, (loss, pred_info) in enumerate(zip(chain_losses, chain_prediction_acc)):
            avg_loss = loss / len(self.eval_loader)
            self.metric_logger.log_metrics(
                {f"cot_eval_loss_chain_idx_{chain_idx}": avg_loss},
                step=step,
            )
            accuracy = pred_info["correct"] / pred_info["total"]
            self.metric_logger.log_metrics(
                {f"cot_eval_accuracy_chain_idx_{chain_idx}": accuracy},
                step=step,
            )

    def create_output_dir(self):
        if self.device_id == 0:
            os.makedirs(os.path.join(self.args.output_base_dir, self.run_name), exist_ok=True)
            # save the args to the output directory
            args_file = os.path.join(self.args.output_base_dir, self.run_name, "args.json")
            with open(args_file, "w") as f:
                json.dump(asdict(self.args), f, indent=4)
            logging.info(f"Saved args to {args_file}")

    def save_model(self):
        if self.device_id == 0:
            model_dir = os.path.join(self.args.output_base_dir, self.run_name, "final_model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            logging.info(f"Saving model to {model_dir}")
            self.model.module.save_pretrained(model_dir)

    def save_checkpoint(self, step: int):
        if self.device_id == 0:
            checkpoint_dir = os.path.join(self.args.output_base_dir, self.run_name, "checkpoints", str(step))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            logging.info(f"Saving checkpoint to {checkpoint_dir}")
            self.model.module.save_pretrained(checkpoint_dir)

    def save_eval_dataset(self):
        if self.device_id == 0:
            eval_dataset_dir = os.path.join(self.args.output_base_dir, self.run_name, "eval_dataset")
            if not os.path.exists(eval_dataset_dir):
                os.makedirs(eval_dataset_dir)
            logging.info(f"Saving eval dataset to {eval_dataset_dir}")
            torch.save(
                self.eval_loader.dataset.data,
                os.path.join(eval_dataset_dir, "eval_data.pt"),
            )


class SpecialTokenTrainer(Trainer):
    def __init__(self, args: Args, device_id: int, world_size: int):
        self.run_name = str(args)
        self.args = args
        self.device_id = device_id
        self.world_size = world_size
        self.train_dataset_clz = TokenizedDataset
        self.eval_dataset_clz = EvalTokenizedDataset
        self.collate_fn = lambda batch: special_token_collate_fn(batch=batch, pad_token_id=args.pad_token_id)
        self.create_output_dir()
        self.create_model()
        self.create_dataloaders()
        self.metric_logger = create_metric_logger()

    @torch.no_grad()
    def cot_evaluate(self, step: int) -> None:
        # Evaluate the model with CoT and without teacher forcing.
        self.model.eval()
        # ensure that we are using special tokens
        assert self.args.reserved_token_ids is not None
        # Initialize answer prediction info
        answer_pred_info = {
            "correct_answer": 0.0,
            "correct_format": 0.0,
            "total": 0.0,
            "num_tokens_generated": [],
        }
        logged_artifact = False
        for batch in self.eval_loader:
            input_ids = batch["cot_eval"]["input_ids"].cuda(self.device_id)
            attention_mask = batch["cot_eval"]["attention_mask"].cuda(self.device_id)
            last_example_cot = batch["cot_eval"]["last_example_cot"].cuda(self.device_id)

            # Generate the chain of tokens using model forward passes without teacher forcing
            # i.e we use the .generate() method.

            generation_config = GenerationConfig(
                max_new_tokens=self.args.max_pred_tokens,
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

            generated_ids = output.sequences  # (B, generated_seq_length)

            # for every response in the batch, obtain the logit of the last non-pad and non-eos token
            # from the generated sequence
            for b in range(generated_ids.shape[0]):
                # skip the input tokens
                output_ids = generated_ids[b][input_ids.shape[-1] :]
                # find the index of the last non-pad and non-eos token
                no_pad_mask = torch.where(output_ids != self.args.pad_token_id)[0]
                output_ids_wo_pad = output_ids[no_pad_mask]
                # We need atleast 4 tokens (answer_start_token_id, ans, answer_end_token_id, eot) to even evaluate.
                # (simulating an instruction following based evaluation)
                if output_ids_wo_pad.numel() < 4:
                    correct_answer = 0
                    correct_format = 0
                elif (
                    output_ids_wo_pad[-2] != self.args.answer_end_token_id
                    or output_ids_wo_pad[-4] != self.args.answer_start_token_id
                ):
                    # if the last token is not the answer_end_token_id or the third last token is not the answer_start_token_id
                    # then we cannot evaluate the answer
                    correct_answer = 0
                    correct_format = 0
                else:
                    # get the label of the last label token
                    # since index -1 will be eos and -2 will be answer_end_token_id we access index -3
                    # (traverse in a reverse manner since padding is applied from the left)
                    label = last_example_cot[b, -3].item()
                    prediction = output_ids_wo_pad[-3].item()
                    assert label not in self.args.reserved_token_ids.values(), (
                        f"Label {label} is a reserved token. Reserved tokens: {self.args.reserved_token_ids}"
                    )
                    # accumulate correct_answer predictions
                    correct_answer = prediction == label
                    correct_format = 1

                # log an output
                if not logged_artifact and self.device_id == 0:
                    temp_dir = tempfile.mkdtemp()
                    artifact_path = os.path.join(temp_dir, f"cot_eval_outputs_{step}.json")
                    with open(artifact_path, "w") as f:
                        json.dump(
                            {
                                "correct_answer": correct_answer,
                                "correct_format": correct_format,
                                "output_ids": output_ids.tolist(),
                                "output_ids_wo_pad": output_ids_wo_pad.tolist(),
                                "last_example_cot": last_example_cot[b].tolist(),
                            },
                            f,
                            indent=4,
                        )
                    self.metric_logger.log_artifact(local_path=artifact_path)
                    logged_artifact = True

                # accumulate scores
                answer_pred_info["correct_answer"] += correct_answer
                answer_pred_info["correct_format"] += correct_format
                answer_pred_info["total"] += 1
                answer_pred_info["num_tokens_generated"].append(output_ids_wo_pad.numel())

        answer_accuracy = answer_pred_info["correct_answer"] / answer_pred_info["total"]
        self.metric_logger.log_metrics(
            {"cot_eval_accuracy_answer_token": answer_accuracy},
            step=step,
        )
        answer_format_accuracy = answer_pred_info["correct_format"] / answer_pred_info["total"]
        self.metric_logger.log_metrics(
            {"cot_eval_accuracy_answer_format": answer_format_accuracy},
            step=step,
        )
        # percentiles of the number of tokens generated
        num_tokens_generated = answer_pred_info["num_tokens_generated"]
        for percentile in [25, 50, 75, 90, 99]:
            value = np.percentile(num_tokens_generated, percentile)
            self.metric_logger.log_metrics(
                {f"cot_eval_num_tokens_generated_percentile_{percentile}": value},
                step=step,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_card_key", type=int, required=True)
    parser_args = parser.parse_args()

    task_card_key = parser_args.task_card_key
    assert task_card_key in TASK_CARD, f"Invalid task_card_key: {task_card_key}"

    args = TASK_CARD[task_card_key]

    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(device_id)
    logging.info(f"=> set cuda device = {device_id}")

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=300))

    trainer_class = SpecialTokenTrainer if args.enable_special_tokens else Trainer
    logging.info(f"Using trainer class: {trainer_class}")

    trainer = trainer_class(
        args=args,
        device_id=device_id,
        world_size=world_size,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_eval_dataset()
