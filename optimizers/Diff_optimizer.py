import torch
import util
from models import MultiGPUModelWrapper
from optimizers.base_optimizer import BaseOptimizer
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class DiffOptimizer(BaseOptimizer):
    """ Class for running the optimization of the model parameters.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--beta1", default=0.9, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)
        parser.add_argument(
            "--R1_once_every", default=16, type=int,
            help="lazy R1 regularization. R1 loss is computed "
                 "once in 1/R1_freq times",
        )
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt
        opt = self.opt
        self.model = model
        self.params = self.model.get_parameters_for_mode("Diff")
        self.optimizer = torch.optim.AdamW(
            self.params, lr=opt.lr, weight_decay=1e-2
        )
        self.scaler = GradScaler()

    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)

    def prepare_images(self, data_i):
        return data_i["real_A"], data_i["mask"]

    def train_one_step(self, data_i, total_steps_so_far):
        images_minibatch, mask_minibatch = self.prepare_images(data_i)
        losses = self.train_generator_one_step(images_minibatch, mask_minibatch)
        return util.to_numpy(losses)

    def train_generator_one_step(self, images, mask):
        self.set_requires_grad(self.params, True)
        self.optimizer.zero_grad()
        g_losses, g_metrics = self.model(
            images, mask, command="compute_generator_losses"
        )        
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer.step()
        g_losses.update(g_metrics)
        return {**g_losses}

    def get_visuals_for_snapshot(self, data_i):
        images, _ = self.prepare_images(data_i)
        with torch.no_grad():
            return self.model(images, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far):
        self.model.save(total_steps_so_far)
