import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import model
from dataset import dataset
from utils import log, fix_seed, make_ep_label, mean_confidence_interval, get_resume_file
from dataset.sampler import CategoriesSampler 

  
if __name__ == "__main__":
    shot = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fix_seed()
    
    config_path = "config/cross-domain_{}-shot_mini-to-cub_wrn.yaml".format(shot)
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    save_dir = os.path.join("save", "{}_{}".format(config["source_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    load_path = get_resume_file(save_dir)
        
    target_dataset = dataset.make(config["target_dataset"]["type"], **config["target_dataset"]["args"], phase="cross_domain")
    target_sampler = CategoriesSampler(target_dataset.labels, config["test_ep"]["batch_num"], config["test_ep"]["n"], config["test_ep"]["k"] + config["test_ep"]["q"]) 
    test_loader = DataLoader(
        target_dataset,
        batch_sampler=target_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    model = model.make(config["model"]["name"], **config["model"]["args"])
    criterion = torch.nn.CrossEntropyLoss()
    log("cross domain {}-shot {} from {} to {}".format(shot, config["model"]["args"]["encoder"]["name"], config["source_dataset"]["name"], config["target_dataset"]["name"]), path="cross_domain")
    # 单给encoder（wrn）加载权重
    model.encoder.load_state_dict(torch.load(load_path)["state"], strict=False)
    
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
        
        log("epoch {}:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(epoch_id, test_loss, np.mean(test_acc) * 100, mean_confidence_interval(test_acc) * 100), path="cross_domain")
    
    final_loss = np.mean(final_loss)
    log("final:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(final_loss, np.mean(final_acc) * 100, mean_confidence_interval(final_acc) * 100), path="cross_domain")