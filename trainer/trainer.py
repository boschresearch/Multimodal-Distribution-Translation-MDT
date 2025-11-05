"""Stripped from https://github.com/wsgharvey/video-diffusion/blob/main/improved_diffusion/train_util.py"""

import copy
import os
import sys
import glob

import blobfile as bf
import torch
import torch as th
from torch.optim import AdamW
from utils import logger
from evaluations.helpers import load_evaluations
from utils.nn import update_ema


class TrainLoop:
    def __init__(
            self,
            mtb,
            train_data,
            test_data,
            batch_size,
            lr,
            ema_rate,
            log_interval,
            test_interval,
            save_interval,
            lr_anneal_steps,
            total_training_steps,
            workdir,
            dataset,
            use_scheduler,
            device,
    ):
        self.mtb = mtb
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = ema_rate
        self.ema_rate = (
            [self.ema_rate] if isinstance(self.ema_rate, float) else [float(x) for x in self.ema_rate.split(",")]
        )
        self.ema_params = [copy.deepcopy(list(self.mtb.parameters())) for _ in range(len(self.ema_rate))]
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.dataset = dataset

        # evaluation object that helps to evaluate different tasks
        self.evaluator = load_evaluations(self.dataset, device)

        self.opt = AdamW(self.mtb.parameters(), lr=self.lr)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 1)
        self.device = device

    def run_loop(self):
        while True:
            for batch, cond, _ in self.train_data:
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                self.run_step(batch, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()

                if self.step % self.test_interval == 0:
                    with torch.no_grad():
                        self.run_test_step(batch, cond)
                        eval_metrics = self.evaluator.evaluate(self.mtb, self.test_data, eval_type='one_batch',
                                                               save_dir=self.workdir)
                        log_score_dict(eval_metrics)
                        self.mtb.train()
                        logger.dumpkvs()

                if self.step % (self.test_interval * 10) == 0:
                    with torch.no_grad():
                        full_eval_metrics = self.evaluator.evaluate(self.mtb, self.test_data, eval_type='full',
                                                                    save_dir=self.workdir)
                        log_score_dict(full_eval_metrics)
                        full_eval_train = self.evaluator.evaluate(self.mtb, copy.deepcopy(self.train_data),
                                                                  eval_type='full', prefix='train')
                        log_score_dict(full_eval_train)
                        self.mtb.train()
                        logger.dumpkvs()

                if self.step % self.save_interval == 0:
                    self.save()
                    pass

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        self._update_ema()
        self._anneal_lr()
        self.log_step()

    def run_test_step(self, batch, cond):
        self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.opt.zero_grad()
        else:
            self.mtb.eval()

        x = batch.to(self.device)
        y = cond.to(self.device)
        model_outputs = self.mtb(x, y)
        model_outputs['x'] = x

        t, weights = self.mtb.bridge_model.schedule_sampler.sample(x.shape[0], self.device)
        loss, loss_dict = self.mtb.loss(model_outputs, self.step)

        loss = (loss * weights).mean()
        log_loss_dict(
            self.mtb.bridge_model.diffusion, t,
            {k if train else 'test_' + k: v * weights for k, v in loss_dict.items()}
        )
        if train:
            loss.backward()
        else:
            self.mtb.train()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mtb.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0] + '*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)

        def save_checkpoint(rate, params):
            state_dict = self.mtb.state_dict()
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model_{(self.step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)

            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        filename = f"opt_{(self.step):06d}.pt"
        if for_preemption:
            filename = f"freq_{filename}"
            maybe_delete_earliest(filename)

        with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mtb.parameters())


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_score_dict(losses):
    for key, values in losses.items():
        logger.logkv(key, values.item())
