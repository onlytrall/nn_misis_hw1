import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import train
import dataset
import wandb
from  models import model_simple



if __name__ ==  "__main__":
  from dotenv import load_dotenv
  load_dotenv()

  wandb.login(key=os.environ["WANDB_API"])

  model = model_simple.Model(output=1, hidden_size=32)
  path_test = 'loan_test.csv'
  path_train = 'loan_train.csv'

  train_dataset = dataset.load_datasets(path_train)
  scaler = train_dataset.scale()
  test_dataset = dataset.load_datasets(path_test)
  test_dataset.scale(scaler)

  config = {
    'lr' : 0.01,
    'epochs' : 100,
    'batch_size' : 256,
    'weight_decay' : 0.01,
    'seed' : 1,
  }

  train.train(train_dataset=train_dataset, eval_dataset=test_dataset, config=config, model=model)