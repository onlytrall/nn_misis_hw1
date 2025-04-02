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