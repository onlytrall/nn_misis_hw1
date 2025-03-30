import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC
from tqdm import tqdm
from dataset import Collator, CustomDataset

def _map_to_device(batch: dict, dev: torch.device) -> dict:
  batch['target'] = batch['target'].to(dev)
  batch['cat_features']['person_home_ownership'] = batch['cat_features']['person_home_ownership'].to(dev)
  batch['cat_features']['loan_intent'] = batch['cat_features']['loan_intent'].to(dev)
  batch['cat_features']['loan_grade'] = batch['cat_features']['loan_grade'].to(dev)
  batch['cat_features']['cb_person_default_on_file'] = batch['cat_features']['cb_person_default_on_file'].to(dev)

  batch['numeric_features']['person_age'] = batch['numeric_features']['person_age'].to(dev)
  batch['numeric_features']['person_income'] = batch['numeric_features']['person_income'].to(dev)
  batch['numeric_features']['person_emp_length'] = batch['numeric_features']['person_emp_length'].to(dev)
  batch['numeric_features']['loan_amnt'] = batch['numeric_features']['loan_amnt'].to(dev)
  batch['numeric_features']['loan_int_rate'] = batch['numeric_features']['loan_int_rate'].to(dev)
  batch['numeric_features']['loan_percent_income'] = batch['numeric_features']['loan_percent_income'].to(dev)
  batch['numeric_features']['cb_person_cred_hist_length'] = batch['numeric_features']['cb_person_cred_hist_length'].to(dev)

def train_step(dev: torch.device, train_dl: DataLoader, loss, optimizer, model):
  model.train()
  train_loss = MeanMetric().to(dev)
  train_rocauc = AUROC(task='binary').to(dev)
  for i, batch in enumerate(train_dl):
    _map_to_device(batch, dev)

    result = model(cat_features=batch['cat_features'], numeric_features=batch['numeric_features'])
    loss_value = loss(result, batch['target'])
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss.update(loss_value)
    train_rocauc.update(torch.sigmoid(result), batch['target'])

  train_loss = train_loss.compute().item()
  train_rocauc = train_rocauc.compute().item()

  return train_loss, train_rocauc

def eval(dev: torch.device, eval_dl: DataLoader, loss, model):
  model.eval()
  eval_loss = MeanMetric().to(dev)
  eval_rocauc = AUROC(task='binary').to(dev)
  with torch.no_grad():
    for i_eval, batch_eval in enumerate(eval_dl):
      _map_to_device(batch_eval, dev)

      result_eval = model(cat_features=batch_eval['cat_features'], numeric_features=batch_eval['numeric_features'])
      eval_loss_value = loss(result_eval, batch_eval['target'])

      eval_loss.update(eval_loss_value)
      eval_rocauc.update(torch.sigmoid(result_eval), batch_eval['target'])
    eval_loss = eval_loss.compute().item()
    eval_rocauc = eval_rocauc.compute().item()
    
    return eval_loss, eval_rocauc
    

def train(train_dataset: Dataset, eval_dataset: Dataset, config: dict, model):
  dev = torch.device('cuda:0')

  lr = config['lr']
  n_epochs = config['epochs']
  batch_size = config['batch_size']
  weight_decay = config['weight_decay']
  seed = config['seed']

  torch.random.manual_seed(seed)

  loss_bce = BCEWithLogitsLoss()

  collator = Collator()
  model = model.to(dev)
  optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

  wandb.init(project="hello-wandb")
  wandb.config = config

  train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)
  eval_dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)

  for i_epoch in tqdm(range(n_epochs)):
    train_loss, train_rocauc = train_step(dev, train_dl, loss_bce, optimizer, model)
    eval_loss, eval_rocauc = eval(dev, eval_dl, loss_bce, model)

    wandb.log({"train_loss" : train_loss, "train_rocauc" : train_rocauc})
    wandb.log({"eval_loss" : eval_loss, "eval_rocauc" : eval_rocauc})

    print(f"train_loss in {i_epoch}: {train_loss}, train_metric: {train_rocauc}") 
    print(f"eval_loss in {i_epoch}: {eval_loss}, eval_metric: {eval_rocauc}")
  

