import sys

sys.path.append('..')
import torch
from ID2S.dataset import load_dataset
from common.trainer import train, val, save_result, save_checkpoint, save_best_checkpoint, load_checkpoint
import os
from dataset_json import load_dataset
from common.loss_function import MAELoss
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model.model import defect_dpp


class Trainer:
    def __init__(self, checkpoint_path, result_path, train_ratio, batch_size, max_epoch, loss_function,
                 model, lr,
                 lr_decay_step_size, lr_decay_factor, device, seed,
                 checkpoint_name, splitting, radius
                 ):

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.train_loader,self.test_loader = load_dataset(train_ratio, seed,batch_size, splitting, radius)
        self.splitting = splitting

        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name

        self.result_path = result_path

        self.device = device
        self.max_epoch = max_epoch
        self.loss_function = loss_function
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        self.start_epoch = 1

    def runner(self):
        print("Start training...")
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            t1 = time.time()
            train_loss = train(data_loader=self.train_loader, model=self.model, optimizer=self.optimizer,
                               device=self.device, loss_function=MAELoss)
            t2 = time.time()
            t = (t2 - t1) / 60
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}: train loss is {train_loss:.4f} eV, learning rate {current_lr:.2e}, cost {t:.1f} minutes")
            save_checkpoint(check_point_path=self.checkpoint_path, optimizer=self.optimizer, scheduler=self.scheduler,
                            model=self.model, epoch=epoch, checkpoint_name=self.checkpoint_name)
            self.scheduler.step()
            torch.cuda.empty_cache()

    def predict(self, checkpoint=None):
        checkpoint = os.path.join(self.checkpoint_path, checkpoint)
        test_model, _, _, _ = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, )
        result = val(self.test_loader, test_model, MAELoss, self.device, test=True)
        return result


if __name__ == '__main__':
    start = time.time()

    seeds = [888, 666, 42, 1234, 66]
    result_path = "result"
    checkpoint_path = 'checkpoint'
    batch_size = 8
    save_results = True
    cutoff = 5
    train_ratio_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    splitting_ways = ["compound", "all"]
    train_methods = ["from-scratch","fine-tune"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for _train in train_methods:
        for splitting in splitting_ways:
            for train_ratio in train_ratio_list:
                for seed in seeds:
                    checkpoint_name = f"{_train}_{train_ratio}_{splitting}_{seed}_checkpoint.pt"
                    result_file = f"{_train}_{train_ratio}_{splitting}_{seed}_result.pkl"
                    model = defect_dpp(hidden_channels=64,
                                       out_emb_channels=128,
                                       out_channels=32,
                                       num_blocks=4,
                                       int_emb_size=64,
                                       basis_emb_size=64,
                                       num_spherical=6,
                                       num_radial=7,
                                       cutoff=cutoff
                                       )
                    if _train == "fine-tune":
                        lr = 0.0002
                        pretrained_model_list = [
                            load_checkpoint(f"../ID2S/checkpoint/best_all_{k}_checkpoint.pt", model) for k in
                            seeds]
                        for _, param in enumerate(model.state_dict()):
                            model.state_dict()[param] = sum(
                                [p_model.state_dict()[param] for p_model in pretrained_model_list]) / 5
                    else:
                        lr = 0.0004
                    model.to(device)
                    run = Trainer(checkpoint_path=checkpoint_path, result_path=result_path,
                                  train_ratio=train_ratio,
                                  batch_size=batch_size,
                                  max_epoch=30, loss_function=MAELoss, model=model, lr=lr, lr_decay_step_size=6,
                                  lr_decay_factor=0.8, device=device, checkpoint_name=checkpoint_name,
                                  seed=seed,
                                  splitting=splitting, radius=cutoff
                                  )

                    print(f"Start {_train}-{train_ratio}-{splitting}-{seed} task")
                    run.runner()
                    result = run.predict(checkpoint=checkpoint_name)
                    result_file = os.path.join(result_path, result_file)
                    save_result(result_path=result_file, result=result)
    end = time.time()
    hour = (end - start) // 3600
    mins = ((end - start) % 3600) // 60
    print(f"Totally cost {int(hour)} hours {int(mins)} minutes.")
