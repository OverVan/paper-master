import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import model
from dataset import dataset
from utils import log, fix_seed, make_optimizer, make_ep_label
from dataset.sampler import CategoriesSampler 

if __name__ == "__main__":
    shot = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fix_seed()
    
    config_path = "config/train_{}-shot_cifar-fs_convnet4.yaml".format(shot)
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    load_dir = os.path.join("save", "{}_{}".format(config["train_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    save_dir = os.path.join("checkpoints", "{}-shot_{}_{}".format(shot, config["train_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    load_path = os.path.join(load_dir, "max-val.pth")
    last_path = os.path.join(save_dir, "last-epoch.pth")
    max_path = os.path.join(save_dir, "max-val.pth")
        
    train_dataset = dataset.make(config["train_dataset"]["type"], **config["train_dataset"]["args"])
    train_sampler = CategoriesSampler(train_dataset.labels, config["train_ep"]["batch_num"], config["train_ep"]["n"], config["train_ep"]["k"] + config["train_ep"]["q"])
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    val_dataset = dataset.make(config["val_dataset"]["type"], **config["val_dataset"]["args"])
    val_sampler = CategoriesSampler(val_dataset.labels, config["val_ep"]["batch_num"], config["val_ep"]["n"], config["val_ep"]["k"] + config["val_ep"]["q"])
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    paper_model = model.make(config["model"]["name"], **config["model"]["args"])
    criterion = torch.nn.CrossEntropyLoss()
    # 接续训练
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path)
        start_epoch = checkpoint["last_epoch"]
        log("continue to train from epoch {}".format(start_epoch + 1))
        paper_model.load_state_dict(checkpoint["model_sd"])
        optimizer = make_optimizer(config["optimizer"]["name"], paper_model.parameters(), **config["optimizer"]["args"])
        optimizer.load_state_dict(checkpoint["optimizer_sd"])
        max_acc = checkpoint["max_acc"]
        best_epoch = checkpoint["best_epoch"]
    else:
        log("train {}-shot {} {}".format(shot, config["train_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
        start_epoch = 0
        pretrained_sd = torch.load(load_path)
        paper_model.load_state_dict(pretrained_sd, strict=False)
        max_acc = 0
        best_epoch = 0
        optimizer = make_optimizer(config["optimizer"]["name"], paper_model.parameters(), **config["optimizer"]["args"])
    
    max_epoch = config["max_epoch"]
    train_label = make_ep_label(config["train_ep"]["n"], config["train_ep"]["q"])
    val_label = make_ep_label(config["val_ep"]["n"], config["val_ep"]["q"])
    for epoch in range(start_epoch, max_epoch):
        epoch_id = epoch + 1
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        paper_model.train()
        for x, _ in train_loader:
            x = x.cuda()
            pred = paper_model(x, config["train_ep"]["n"], config["train_ep"]["k"], config["train_ep"]["q"])
            loss = criterion(pred, train_label)
            acc = (torch.argmax(pred, dim=1) == train_label).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            
        paper_model.eval()
        for x, _ in val_loader:
            x = x.cuda()
            with torch.no_grad():
                pred = paper_model(x, config["val_ep"]["n"], config["val_ep"]["k"], config["val_ep"]["q"])
                loss = criterion(pred, val_label)
                acc = (torch.argmax(pred, dim=1) == val_label).float().mean()
                
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)
        
        log("epoch {}:\n\ttrain loss: {:.4f}\ttrain acc: {:.2%}\n\tval loss: {:.4f}\tval acc: {:.2%}".format(epoch_id, train_loss, train_acc, val_loss, val_acc))
        
        if val_acc >= max_acc:
            max_acc = val_acc
            best_epoch = epoch_id
            torch.save(paper_model.state_dict(), max_path)
        log("\tbest epoch: {}\tmax acc: {:.2%}".format(best_epoch, max_acc))
        checkpoint = {
            "last_epoch": epoch_id,
            "model_sd": paper_model.state_dict(),
            "optimizer_sd": optimizer.state_dict(),
            "max_acc": max_acc,
            "best_epoch": best_epoch
        }
        torch.save(checkpoint, last_path)