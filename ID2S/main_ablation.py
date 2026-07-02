import sys
sys.path.append('..')
import torch
from common.trainer import train, val, save_result, save_checkpoint, save_best_checkpoint, load_checkpoint
import os
from dataset import load_dataset
from common.loss_function import MAELoss
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model.model import defect_dpp
from model.Other_models.MACE import MACE
from model.Other_models.DimeNetplusplus import DimeNetPlusPlus
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Trainer:
    def __init__(self, checkpoint_path, result_path,train_ratio, valid_ratio, batch_size, vt_batch_size,cutoff,
                 max_epoch, loss_function, model, lr, lr_decay_step_size, lr_decay_factor, device, seed, checkpoint_name,splitting,
                 checkpoint=None):

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.train_loader, self.valid_loader, self.test_loader = load_dataset(train_ratio, valid_ratio, seed,
                                                                              batch_size, splitting,radius=cutoff)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name=checkpoint_name
        self.splitting = splitting
        self.result_path = result_path
        self.vt_batch_size = vt_batch_size
        self.device = device

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
        self.train_loss = []
        self.valid_loss = []
    def runner(self, num_epochs_report_loss):
        print("Start training...")
        best_mae = 1e9
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            t1 = time.time()
            train_loss = train(data_loader=self.train_loader, model=self.model, optimizer=self.optimizer,
                               device=self.device,
                               loss_function=MAELoss)
            torch.cuda.empty_cache()
            valid_loss = val(self.valid_loader, self.model, MAELoss, self.device, test=False)
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
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
            if epoch % num_epochs_report_loss == 0:
                test_result = val(self.test_loader, self.model, MAELoss, self.device, test=True)
                test_loss = np.mean(abs(test_result["truth"] - test_result["prediction"]))
                print(f"Test loss is {test_loss:.4f} eV")

    def predict(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = os.path.join(self.checkpoint_path, f"{self.splitting}_best_checkpoint.pt")
        else:
            checkpoint = os.path.join(self.checkpoint_path, checkpoint)
        test_model, _, _, _ = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, )
        result = val(self.test_loader, test_model, MAELoss, self.device, test=True)
        return result


if __name__ == '__main__':
    import pickle
    # model_name="MACE"
    # seeds = [888, 1234, 666, 18]
    seeds=[18]
    start = time.time()
    lr = 0.0004

    dataset_path = 'data'
    batch_size = 4
    save_results = True
    cutoff=5
    splitting_list = ["compound","all"]
    # for contribution in [["V"],["G"]]:
    for contribution in [["V","G"]]:
        result_path = f"result_{contribution}"
        checkpoint_path = f'checkpoint_{contribution}'
        for seed in seeds:
            # if contribution == ["V"]:
            #     if seed == 888 or seed==1234:
            #         continue

            for splitting in splitting_list:
                checkpoint_name = f"{splitting}_{seed}_checkpoint.pt"
                result_file = f"{splitting}_{seed}_result.pkl"


                model = defect_dpp(hidden_channels=64,
                                       out_emb_channels=128,
                                       out_channels=32,
                                       num_blocks=4,
                                       int_emb_size=64,
                                       basis_emb_size=64,
                                       num_spherical=6,
                                       num_radial=7,
                                    cutoff=cutoff,contribution=contribution)

                num_params = sum(p.numel() for p in model.parameters())

                print(f"Loading model with {num_params} parameters")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                run = Trainer(checkpoint_path=checkpoint_path, result_path=result_path,  train_ratio=0.8,
                              valid_ratio=0.1, batch_size=batch_size, vt_batch_size=16,
                              max_epoch=50, loss_function=MAELoss, model=model, lr=lr, lr_decay_step_size=5,
                              lr_decay_factor=0.8, device=device, checkpoint_name=checkpoint_name,splitting=splitting,seed=seed,cutoff=cutoff)

                print(f"Start {splitting} task")
                run.runner(num_epochs_report_loss=100)
                result = run.predict("best_"+checkpoint_name)
                result_file = os.path.join(result_path, result_file)
                save_result(result_path=result_file, result=result)
                with open("train_curve.pkl", "wb") as f:
                    cures={"train":run.train_loss, "valid":run.valid_loss}
                    pickle.dump(cures, f)
    end = time.time()
    hour = (end - start) // 3600
    mins = ((end - start) % 3600) // 60
    print(f"Totally cost {int(hour)} hours {int(mins)} minutes.")
