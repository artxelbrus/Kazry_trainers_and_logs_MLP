import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Muon, AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm
import os
import random
import numpy as np
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class KazryPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.where(x>=0.0, x, torch.exp(x)*x)
class HIGGSDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.data = pd.read_parquet(dataset_path)
    def __len__(self):
        return len(self.data.iloc[:, 0])
    def __getitem__(self, index):
        data_idx = torch.from_numpy(self.data.iloc[index, :].to_numpy())
        return data_idx[1:], data_idx[0]
class FFNLayer(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim, bias=False),
            activation()
        )
        nn.init.normal_(self.layer[1].weight, std=0.02)
    def forward(self, x):
        return self.layer(x) + x
class Model(nn.Module):
    def __init__(self, dim, layers, activation):
        super().__init__()
        self.emb = nn.Linear(28, dim, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(FFNLayer(dim, activation))
        self.final_norm = nn.RMSNorm(dim)
        self.unemb = nn.Linear(dim, 1, bias=False)
        nn.init.normal_(self.unemb.weight, std=0.02)
    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.unemb(self.final_norm(x))
def train(
    learning_rate: float = 1e-3,
    batch_size: int = 8192,
    weight_decay: float = 1e-3,
    epochs: int = 15,
    shuffle: bool = True,
    num_workers: int = 8,
    train_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    checkpoint_dir: str | None = None,
    log_dir: str | None = None,
    activation = KazryPyTorch,
    dim: int = 64,
    layers: int = 8
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    dataset_train = HIGGSDataset(train_dataset_path)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    dataset_val = HIGGSDataset(val_dataset_path)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    model = Model(dim, layers, activation).to("cuda")
    model = torch.compile(model=model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    muon_weights = []
    adamw_weights = []
    loss_val = []
    loss_train = []
    for p in model.layers.parameters():
        if p.ndim == 2:
            muon_weights.append(p)
        else:
            adamw_weights.append(p)
    adamw_weights.extend(model.emb.parameters())
    adamw_weights.extend(model.final_norm.parameters())
    adamw_weights.extend(model.unemb.parameters())
    muon_optim = Muon(muon_weights, learning_rate, weight_decay=weight_decay)
    adamw_optim = AdamW(adamw_weights, learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    metric = tm.AUROC("binary").to("cuda")
    sch_adam = lr_scheduler.CosineAnnealingLR(adamw_optim, T_max=epochs*len(dataloader_train), eta_min=1e-5)
    sch_muon = lr_scheduler.CosineAnnealingLR(muon_optim, T_max=epochs*len(dataloader_val), eta_min=1e-5)
    for epoch in range(epochs):
        model.train()
        for batch_index, (input_to_model, target) in enumerate(dataloader_train):
            input_to_model, target = input_to_model.to("cuda", torch.bfloat16, non_blocking=True), target.to("cuda", torch.bfloat16, non_blocking=True).unsqueeze(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_to_model)
                loss = criterion(output, target)
                muon_optim.zero_grad()
                adamw_optim.zero_grad()
                loss.backward()
                muon_optim.step()
                adamw_optim.step()
                loss_train.append(loss)
                sch_adam.step()
                sch_muon.step()
        print(f"{sum(loss_train)/len(loss_train)} train epoch: {epoch}")
        writer.add_scalar("loss/cross entropy train", scalar_value=sum(loss_train)/len(loss_train), global_step=epoch)
        loss_train.clear()
        model.eval()
        for batch_index, (input_to_model, target) in enumerate(dataloader_val):
            input_to_model, target = input_to_model.to("cuda", torch.bfloat16), target.to("cuda", torch.bfloat16).unsqueeze(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                output = model(input_to_model)
                loss = criterion(output, target)
                metric.update(output, target)
                loss_val.append(loss)
        print(f"{sum(loss_val)/len(loss_val)} val epoch: {epoch}")
        writer.add_scalar("loss/cross entropy val", scalar_value=sum(loss_val)/len(loss_val), global_step=epoch)
        loss_val.clear()
        auc = metric.compute()
        metric.reset()
        writer.add_scalar("loss/auc val", scalar_value=auc, global_step=epoch)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pth"))
if __name__ == "__main__":
    seed_everything()
    train(train_dataset_path="HIGGS_train.parquet", val_dataset_path="HIGGS_val.parquet", checkpoint_dir="kazry/checkpoints", log_dir="kazry/logs")