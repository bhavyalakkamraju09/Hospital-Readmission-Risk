import pandas as pd, numpy as np, os
np.random.seed(42)
N = 30000
ages = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
age_w = [0.003,0.008,0.018,0.035,0.08,0.15,0.22,0.27,0.18,0.036]
number_inpatient = np.random.negative_binomial(1,0.6,N).clip(0,21)
number_emergency = np.random.poisson(0.3,N).clip(0,10)
num_medications  = np.random.poisson(16,N).clip(1,81)
time_in_hospital = np.random.choice(range(1,15),N,p=[0.12,0.14,0.14,0.12,0.10,0.09,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.01])
number_diagnoses = np.random.choice(range(1,10),N,p=[0.03,0.08,0.12,0.14,0.16,0.17,0.14,0.10,0.06])
a1c    = np.random.choice(['>8','>7','Norm','None'],N,p=[0.09,0.02,0.06,0.83])
change = np.random.choice(['No','Ch'],N,p=[0.54,0.46])
insulin = np.random.choice(['No','Steady','Up','Down'],N,p=[0.46,0.30,0.14,0.10])
admission_type = np.random.choice([1,2,3,4,5,6,7,8],N,p=[0.45,0.20,0.15,0.05,0.04,0.04,0.04,0.03])
risk = (number_inpatient*0.55+(num_medications>15)*0.45+(time_in_hospital>7)*0.38+
        (number_emergency>0)*0.35+(a1c=='>8')*0.40+(change=='Ch')*0.25+
        (number_diagnoses>7)*0.35+(admission_type==1)*0.22+(insulin!='No')*0.20+
        num_medications*0.02+number_emergency*0.30+np.random.normal(0,0.5,N))
prob = 1/(1+np.exp(-(risk-2.0)))
readmitted_30 = (prob > np.random.rand(N)).astype(int)
icd9 = ['250.00','250.01','250.40','414.01','428.0','401.9','486','491.21','585.3','584.9','276.1','285.9','272.4','038.9','410.01']
df = pd.DataFrame({
    'encounter_id': range(1000000,1000000+N),
    'patient_nbr': np.random.randint(1,int(N*0.85),N),
    'race': np.random.choice(['Caucasian','AfricanAmerican','Hispanic','Asian','Other','?'],N,p=[0.74,0.19,0.02,0.006,0.02,0.024]),
    'gender': np.random.choice(['Male','Female'],N),
    'age': np.random.choice(ages,N,p=age_w),
    'weight': '?', 'admission_type_id': admission_type,
    'discharge_disposition_id': np.random.choice(range(1,11),N,p=[0.55,0.10,0.08,0.04,0.04,0.03,0.03,0.05,0.04,0.04]),
    'admission_source_id': np.random.choice(range(1,9),N,p=[0.35,0.05,0.05,0.28,0.10,0.05,0.08,0.04]),
    'time_in_hospital': time_in_hospital, 'payer_code': 'MC',
    'medical_specialty': np.random.choice(['InternalMedicine','Emergency/Trauma','Cardiology','Nephrology','?'],N),
    'num_lab_procedures': np.random.poisson(43,N).clip(1,132),
    'num_procedures': np.random.poisson(1.3,N).clip(0,6),
    'num_medications': num_medications,
    'number_outpatient': np.random.poisson(0.37,N).clip(0,15),
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'diag_1': np.random.choice(icd9,N),
    'diag_2': np.random.choice(icd9+['?'],N),
    'diag_3': np.random.choice(icd9+['?'],N),
    'metformin': np.random.choice(['No','Steady','Up','Down'],N,p=[0.52,0.32,0.09,0.07]),
    'repaglinide': 'No', 'nateglinide': 'No', 'glimepiride': 'No',
    'glipizide': np.random.choice(['No','Steady','Up','Down'],N,p=[0.62,0.25,0.07,0.06]),
    'glyburide': np.random.choice(['No','Steady','Up','Down'],N,p=[0.68,0.22,0.05,0.05]),
    'insulin': insulin, 'change': change,
    'diabetesMed': np.random.choice(['Yes','No'],N,p=[0.77,0.23]),
    'A1Cresult': a1c,
    'readmitted': np.where(readmitted_30,'<30',np.where(np.random.rand(N)>0.5,'NO','>30')),
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/diabetic_data.csv', index=False)
print(f'Done: {df.shape} | readmit rate: {(df.readmitted=="<30").mean():.1%}')
