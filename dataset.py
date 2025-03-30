from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

_numeric_feaures = [
  'person_age',
  'person_income',
  'person_emp_length',
  'loan_amnt',
  'loan_int_rate',
  'loan_percent_income',
  'cb_person_cred_hist_length',
]

_person_home_ownership_map = {
  'MORTGAGE' : 0,
  'OTHER' : 1,
  'OWN' : 2,
  'RENT' : 3
}

_loan_intent_map = {
  'DEBTCONSOLIDATION' : 0,
  'EDUCATION' : 1,
  'HOMEIMPROVEMENT' : 2,
  'MEDICAL' : 3,
  'PERSONAL' : 4,
  'VENTURE' : 5
}

_loan_grade_map = {
  'A' : 0,
  'B' : 1,
  'C' : 2,
  'D' : 3,
  'E' : 4,
  'F' : 5,
  'G' : 6,
}

_cb_person_default_on_file_map = {
  'N' : 0,
  'Y' : 1,
}

class CustomDataset(Dataset):
  def __init__(self, df):
    super().__init__()
    self.data = df

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    row = self.data.iloc[index]

    cat_features = {
      'person_home_ownership' :  torch.scalar_tensor(_person_home_ownership_map[row['person_home_ownership']], dtype=torch.long),
      'loan_intent' :  torch.scalar_tensor(_loan_intent_map[row['loan_intent']], dtype=torch.long),
      'loan_grade' :  torch.scalar_tensor(_loan_grade_map[row['loan_grade']], dtype=torch.long),
      'cb_person_default_on_file' :  torch.scalar_tensor(_cb_person_default_on_file_map[row['cb_person_default_on_file']], dtype=torch.long),
    },

    numeric_features = {
      'person_age' : torch.scalar_tensor(row['person_age'], dtype=torch.float32),
      'person_income' : torch.scalar_tensor(row['person_income'], dtype=torch.float32),
      'person_emp_length' : torch.scalar_tensor(row['person_emp_length'], dtype=torch.float32),
      'loan_amnt' : torch.scalar_tensor(row['loan_amnt'], dtype=torch.float32),
      'loan_int_rate' : torch.scalar_tensor(row['loan_int_rate'], dtype=torch.float32),
      'loan_percent_income' : torch.scalar_tensor(row['loan_percent_income'], dtype=torch.float32),
      'cb_person_cred_hist_length' : torch.scalar_tensor(row['cb_person_cred_hist_length'], dtype=torch.float32),
    }
    
    return {
      'target' : torch.scalar_tensor(row['loan_status'], dtype=float),
      'cat_features' : cat_features,
      'numeric_features' : numeric_features,
    }
  
  def scale(self, scaler: MinMaxScaler = None):
    if scaler is None:
      scaler = MinMaxScaler()
      scaler.fit(X=self.data[_numeric_feaures])
    self.data[_numeric_feaures] = scaler.transform(X=self.data[_numeric_feaures])

    return scaler


class Collator:
  def __call__(self, items):
    return {
      'target': torch.stack([x['target'] for x in items]),
      'cat_features' : {
        'person_home_ownership' :  torch.stack([x['cat_features'][0]['person_home_ownership'] for x in items]),
        'loan_intent' :  torch.stack([x['cat_features'][0]['loan_intent'] for x in items]),
        'loan_grade' :  torch.stack([x['cat_features'][0]['loan_grade'] for x in items]),
        'cb_person_default_on_file' :  torch.stack([x['cat_features'][0]['cb_person_default_on_file'] for x in items]),
      },

      'numeric_features' : {
        'person_age' : torch.stack([x['numeric_features']['person_age'] for x in items]),
        'person_income' : torch.stack([x['numeric_features']['person_income'] for x in items]),
        'person_emp_length' : torch.stack([x['numeric_features']['person_emp_length'] for x in items]),
        'loan_amnt' : torch.stack([x['numeric_features']['loan_amnt'] for x in items]),
        'loan_int_rate' : torch.stack([x['numeric_features']['loan_int_rate'] for x in items]),
        'loan_percent_income' : torch.stack([x['numeric_features']['loan_percent_income'] for x in items]),
        'cb_person_cred_hist_length' : torch.stack([x['numeric_features']['cb_person_cred_hist_length'] for x in items]),
      },
    }
  
def load_datasets(path):
  df = pd.read_csv(path)
  return CustomDataset(df)