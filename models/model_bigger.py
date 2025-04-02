import torch
from torch import nn, Tensor

from utils.mapping import (
  _person_home_ownership_map,
  _loan_intent_map,
  _numeric_feaures,
  _cb_person_default_on_file_map,
  _loan_grade_map,
)

_NUMERICAL_FEATURES = len(_numeric_feaures)

class Block(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()

    self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
    self.relu1 = nn.LeakyReLU()
    self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)

    return x

class Model(nn.Module):
  def __init__(self, output, hidden_size):
    super().__init__()
    
    self.emb1 = nn.Embedding(len(_person_home_ownership_map), hidden_size)
    self.emb2 = nn.Embedding(len(_loan_intent_map), hidden_size)
    self.emb3 = nn.Embedding(len(_loan_grade_map), hidden_size)
    self.emb4 = nn.Embedding(len(_cb_person_default_on_file_map), hidden_size)

    self.num_linear = nn.Linear(_NUMERICAL_FEATURES, hidden_size)

    self.block1 = Block(hidden_size)
    self.block2 = Block(hidden_size)
    self.block3 = Block(hidden_size)

    self.linear_cl = nn.Linear(hidden_size, output)

  def forward(self, cat_features, numeric_features):
    x1 = self.emb1(cat_features['person_home_ownership'])
    x2 = self.emb2(cat_features['loan_intent'])
    x3 = self.emb3(cat_features['loan_grade'])
    x4 = self.emb4(cat_features['cb_person_default_on_file'])

    stacked_numeric = torch.stack([
      numeric_features['person_age'], 
      numeric_features['person_income'], 
      numeric_features['person_emp_length'], 
      numeric_features['loan_amnt'], 
      numeric_features['loan_int_rate'], 
      numeric_features['cb_person_cred_hist_length'], 
      numeric_features['loan_percent_income']], 
      dim=-1
    )

    x_num = self.num_linear(stacked_numeric)

    x = x1 + x2 + x3 + x4 + x_num

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)

    x = self.linear_cl(x)
    x = x.squeeze(-1)

    return x
