import torch
import os
import pickle

from sklearn.externals.array_api_compat.torch import reshape

from common.loss_function import weightedMAELoss
from tqdm import tqdm


def train(data_loader, model, loss_function, optimizer, device):
    loss_accum = 0
    step = None
    model.train()
    for step, data in enumerate(tqdm(data_loader,disable=True)):
        data = data.to(device)
        out = model(data)
        if out.shape!=data.y.shape:
            out = reshape(out,data.y.shape)
        # loss = loss_function(out, data)
        if loss_function == weightedMAELoss:
            loss = loss_function(out, data.y, weights=data.weights, reduction="mean")
        else:
            loss = loss_function(out, data.y, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_accum = loss.detach()+loss_accum
    return loss_accum.detach().cpu().numpy() / (step + 1)


def val(data_loader, model, loss_function, device, test):
    model.eval()
    with torch.no_grad():
        if not test:
            loss_accum = 0
            step = None
            model.eval()
            for step, data in enumerate(data_loader):
                data = data.to(device)
                out = model(data)
                if out.shape != data.y.shape:
                    out = reshape(out, data.y.shape)
                if loss_function == weightedMAELoss:
                    loss = loss_function(out, data.y, weights=data.weights, reduction="mean")
                else:
                    loss = loss_function(out, data.y, reduction="mean")
                loss_accum += loss.detach().cpu().item()
            return loss_accum / (step + 1)
        if test:
            truth = torch.tensor([])
            prediction = torch.tensor([])
            defect_type=[]

            for step, data in enumerate(data_loader):
                data = data.to(device)
                out = model(data)

                truth = torch.concat([truth, data.y.cpu()])
                prediction = torch.concat([prediction, out.detach().cpu()])
                if hasattr(data, "defect_type"):
                    defect_type+=data.defect_type

            result = {
                    "truth": truth.numpy(),
                    "prediction": prediction.numpy(),
                    "defect_type": defect_type
                }
            return result


def save_checkpoint(check_point_path, optimizer, scheduler, model, epoch, checkpoint_name):
    check_point_file = os.path.join(check_point_path, checkpoint_name)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'scheduler': scheduler.state_dict()}
    torch.save(state, check_point_file)


def save_best_checkpoint(check_point_path, optimizer, scheduler, model, epoch, metrics, best_metrics, higher_better,
                         checkpoint_name):
    check_point_file = os.path.join(check_point_path, f"best_{checkpoint_name}")
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'scheduler': scheduler.state_dict()}
    if higher_better:
        if metrics > best_metrics:
            best_metrics = metrics
            torch.save(state, check_point_file)
    else:
        if metrics < best_metrics:
            best_metrics = metrics
            torch.save(state, check_point_file)
    return best_metrics


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        return model, optimizer, scheduler, start_epoch
    if scheduler is None and optimizer is None:
        return model


def save_result(result_path, result):
    with open(result_path, 'wb') as F:
        pickle.dump(result, F)
