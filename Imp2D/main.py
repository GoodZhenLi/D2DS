import sys

sys.path.append('..')
import torch

from common.trainer import train, val, save_result, save_checkpoint, save_best_checkpoint, load_checkpoint
import os
from dataset import load_dataset
from common.loss_function import weightedMAELoss, MAELoss
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model.model import defect_dpp


class Trainer:
    def __init__(self, checkpoint_path, result_path, dataset, train_ratio, valid_ratio, batch_size, vt_batch_size,
                 max_epoch, loss_function, model, lr, lr_decay_step_size, lr_decay_factor, device, seed,cutoff,
                 checkpoint_name,
                 checkpoint=None):

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.train_loader, self.valid_loader, self.test_loader = load_dataset(seed, batch_size, train_ratio,valid_ratio,cutoff=cutoff)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.result_path = result_path
        self.vt_batch_size = vt_batch_size
        self.device = device
        self.dataset = dataset
        self.max_epoch = max_epoch
        self.loss_function = loss_function
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        self.start_epoch = 1
        if checkpoint is not None:
            checkpoint = os.path.join(checkpoint_path, checkpoint)
            self.model, self.optimizer, self.scheduler, start_epoch = load_checkpoint(checkpoint,
                                                                                      self.model, self.optimizer,
                                                                                      self.scheduler,
                                                                                      )

    def runner(self, ):
        print("Start training...")
        best_mae = 1e9
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            t1 = time.time()
            train_loss = train(data_loader=self.train_loader, model=self.model, optimizer=self.optimizer,
                               device=self.device,
                               loss_function=weightedMAELoss if self.dataset == "dataset1" else MAELoss)
            torch.cuda.empty_cache()
            valid_loss = val(self.valid_loader, self.model, MAELoss, self.device, test=False)
            t2 = time.time()
            t = (t2 - t1) / 60
            current_lr = self.optimizer.param_groups[0]['lr']
            best_mae = save_best_checkpoint(self.checkpoint_path, self.optimizer, self.scheduler, self.model, epoch,
                                            metrics=valid_loss, best_metrics=best_mae, higher_better=False,
                                            checkpoint_name=self.checkpoint_name)
            print(
                f"Epoch {epoch}: train loss is {train_loss:.4f} eV, valid loss is {valid_loss:.4f} eV, best val loss {best_mae:.4f} eV, learning rate {current_lr:.2e}, cost {t:.1f} minutes")
            save_checkpoint(check_point_path=self.checkpoint_path, optimizer=self.optimizer, scheduler=self.scheduler,
                            model=self.model, epoch=epoch, checkpoint_name=self.checkpoint_name)
            self.scheduler.step()
            torch.cuda.empty_cache()

    def predict(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = os.path.join(self.checkpoint_path, f"{self.splitting}_best_checkpoint.pt")
        else:
            checkpoint = os.path.join(self.checkpoint_path, checkpoint)
        test_model, _, _, _ = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, )
        result = val(self.test_loader, test_model, MAELoss, self.device, test=True)
        return result


if __name__ == '__main__':
    start = time.time()
    lr = 0.0004
    seed = 42
    cutoff=5
    result_path = "result"
    checkpoint_path = 'checkpoint'
    dataset_path = 'data'
    batch_size = 2
    dataset = "database.json"
    checkpoint_name = f"all_imp2d_checkpoint.pt"
    result_file = f"imp2d_result.pkl"
    model = defect_dpp(hidden_channels=128,
                       out_emb_channels=256,
                       out_channels=128,
                       num_blocks=4,
                       int_emb_size=128,
                       basis_emb_size=64,
                       num_spherical=6,
                       num_radial=7,
                       cutoff=cutoff,
                       )
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Loading model with {num_params} parameters")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run = Trainer(checkpoint_path=checkpoint_path, result_path=result_path, dataset=dataset, train_ratio=0.8,
                  valid_ratio=0.1, batch_size=batch_size, vt_batch_size=16,
                  max_epoch=60, loss_function=MAELoss, model=model, lr=lr, lr_decay_step_size=5,
                  lr_decay_factor=0.8, device=device, checkpoint_name=checkpoint_name, seed=seed,cutoff=cutoff)
    run.runner()
    result = run.predict("best_"+checkpoint_name)
    result_file = os.path.join(result_path, result_file)
    save_result(result_path=result_file, result=result)
    end = time.time()
    hour = (end - start) // 3600
    mins = ((end - start) % 3600) // 60
    print(f"Totally cost {int(hour)} hours {int(mins)} minutes.")
