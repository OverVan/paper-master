import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import dataset
from model import model
from utils import log, fix_seed, make_ep_label, mean_confidence_interval
from dataset.sampler import CategoriesSampler 

if __name__ == "__main__":
    shot = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fix_seed()
    
    config_path = "config/test_{}-shot_mini_convnet4.yaml".format(shot)
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    save_dir = os.path.join("checkpoints", "{}-shot_{}_{}".format(shot, config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    max_path = os.path.join(save_dir, "max-val.pth")
        
    test_dataset = dataset.make(config["test_dataset"]["type"], **config["test_dataset"]["args"], phase="test")
    test_sampler = CategoriesSampler(test_dataset.labels, config["test_ep"]["batch_num"], config["test_ep"]["n"], config["test_ep"]["k"] + config["test_ep"]["q"]) 
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    paper_model = model.make(config["model"]["name"], **config["model"]["args"])
    criterion = torch.nn.CrossEntropyLoss()
    log("test {}-shot {} {}".format(shot, config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]), path="test")
    paper_model.load_state_dict(torch.load(max_path))
    
    max_epoch = config["max_epoch"]
    test_label = make_ep_label(config["test_ep"]["n"], config["test_ep"]["q"])
    final_loss = []
    final_acc = []
    for epoch in range(max_epoch):
        epoch_id = epoch + 1
        test_loss = []
        test_acc = []
        
        paper_model.eval()
        for x, _ in test_loader:
            x = x.cuda()
            with torch.no_grad():
                pred = paper_model(x, config["test_ep"]["n"], config["test_ep"]["k"], config["test_ep"]["q"])
                loss = criterion(pred, test_label)
                acc = (torch.argmax(pred, dim=1) == test_label).float().mean()
                
            test_loss.append(loss.item())
            test_acc.append(acc.item())
            
        test_loss = np.mean(test_loss)
        final_loss.append(test_loss)
        final_acc.append(np.mean(test_acc))
        
        log("epoch {}:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(epoch_id, test_loss, np.mean(test_acc) * 100, mean_confidence_interval(test_acc) * 100), path="test")
    
    final_loss = np.mean(final_loss)
    log("final:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(final_loss, np.mean(final_acc) * 100, mean_confidence_interval(final_acc) * 100), path="test")