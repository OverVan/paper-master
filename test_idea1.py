import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.model import make
from dataset.simple_dataset import SimpleDataset
from utils import log, fix_seed, make_ep_label, mean_confidence_interval
from dataset.my_sampler import CategoriesSampler 

if __name__ == "__main__":
    shot = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fix_seed()
    
    config_path = "config/test_idea1_{}-shot_mini_resnet12.yaml".format(shot)
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    save_dir = os.path.join("idea1_checkpoints", "{}-shot_{}_{}".format(shot, config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    max_path = os.path.join(save_dir, "max-val.pth")
        
    test_dataset = SimpleDataset(**config["test_dataset"]["args"], phase="test_idea1")
    test_sampler = CategoriesSampler(test_dataset.labels2inds, config["test_ep"]["batch_num"], config["test_ep"]["n"], config["test_ep"]["k"] + config["test_ep"]["q"])
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    model = make(config["model"]["name"], **config["model"]["args"])
    criterion = torch.nn.CrossEntropyLoss()
    log("test {}-shot {} {}".format(shot, config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]), path="test_idea1")
    model.load_state_dict(torch.load(max_path))
    
    max_epoch = config["max_epoch"]
    test_label = make_ep_label(config["test_ep"]["n"], config["test_ep"]["q"])
    final_loss = []
    final_acc = []
    for epoch in range(max_epoch):
        epoch_id = epoch + 1
        test_loss = []
        test_acc = []
        
        model.eval()
        for x, _ in test_loader:
            x = x.cuda()
            with torch.no_grad():
                pred = model(x, config["test_ep"]["n"], config["test_ep"]["k"], config["test_ep"]["q"])
                loss = criterion(pred, test_label)
                acc = (torch.argmax(pred, dim=1) == test_label).float().mean()
                
            test_loss.append(loss.item())
            test_acc.append(acc.item())
            
        test_loss = np.mean(test_loss)
        final_loss.append(test_loss)
        final_acc.append(np.mean(test_acc))
        
        log("epoch {}:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(epoch_id, test_loss, np.mean(test_acc) * 100, mean_confidence_interval(test_acc) * 100), path="test_idea1")
    
    final_loss = np.mean(final_loss)
    log("final:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(final_loss, np.mean(final_acc) * 100, mean_confidence_interval(final_acc) * 100), path="test_idea1")