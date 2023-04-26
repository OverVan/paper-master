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
    
    config_path = "config/test_no_{}-shot_mini_wrn.yaml".format(shot)
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.load(file, yaml.FullLoader)
        
    save_dir = os.path.join("save", "{}_{}".format(config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]))
    load_path = get_resume_file(save_dir)
        
    test_dataset = dataset.make(config["test_dataset"]["type"], **config["test_dataset"]["args"], phase="infer")
    test_sampler = CategoriesSampler(test_dataset.labels, config["test_ep"]["batch_num"], config["test_ep"]["n"], config["test_ep"]["k"] + config["test_ep"]["q"]) 
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    paper_model = model.make(config["model"]["name"], **config["model"]["args"])
    criterion = torch.nn.CrossEntropyLoss()
    log("infer {}-shot {} {}".format(shot, config["test_dataset"]["name"], config["model"]["args"]["encoder"]["name"]), path="infer")
    # 单给encoder（wrn）加载权重
    # print(model.encoder.state_dict().keys())
    paper_model.encoder.load_state_dict(torch.load(load_path)["state"], strict=False)
    
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
        
        log("epoch {}:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(epoch_id, test_loss, np.mean(test_acc) * 100, mean_confidence_interval(test_acc) * 100), path="infer")
    
    final_loss = np.mean(final_loss)
    log("final:\n\tloss: {:.4f}\tacc: {:.2f} +- {:.2f} (%)".format(final_loss, np.mean(final_acc) * 100, mean_confidence_interval(final_acc) * 100), path="infer")