# MRSC

This is the implementation for paper "Multi-hop Reading on Memory Neural Network with Selective Coverage for Medication Recommendation"

This paper proposes the Multi-hop Reading with Selective Coverage (MRSC) for medication recommendation. MRSC applies covearge model to the attentive multi-hop reading on Memory Neural Network (MemNN), which is built based on Electronic Healthcare Records (EHRs) of patients, and reasons over selected information from MemNN to derive informative patterns for medication recommendation. MRSC encodes temporal patterns contained in EHRs with GRU to obtain the representation of each admission to the hospital, and uses these representations as well as corresponding medications to build a key-value MemNN. For patients without enough admissions, a regular multi-hop reading is conducted to recommend medications. Meanwhile, for those patients with enough admissions, MRSC firstly carries out regular attentive multi-hop reading on MemNN, and keeps track of the coverage information during this process. Note that the coverage model is not integrated into the reading process. Then MRSC selects important information from MemNN for the subsequent reading, and two different selection strategy are proposed, i.e., hard selection and soft selection. After that, the coverage model will be combined with the attentive reading, which helps balance attentions over selected information to avoid over-reading or under-reading. MRSC applies two different coverage strategies, i.e., sum based coverage and GRU based coverage.

# Requirement
Pytorch 1.1

Python 3.7

# Data
Experiments are conducted based on [MIMIC-III](https://mimic.physionet.org), a real-world clinical dataset that collects EHRs related to over 45,000 patients. MRSC uses diagnoses and procedures as inputs, and the goal is to recommend medications prescribed by patients in the first 24-hour of admissions.

After builiding vocabularies for diagnoses, procedures, and medications, medical records are represented by medical codes in the form of integers. Each admission contains three lists of integers, which are diagnoses, procedures, and medications respectively. All admissions of a patient are sorted chronologically to form a list. 

Here is an example of the data. There are two admissions for the patient. For the first one, the patient is diagnosed with diseases 4, 3, 6 and 5, and procedures 34, 2, 35 and 3 as well as four drugs, including 8, 75, 6 and 34 are prescribed. Similarly, the second admission contains diseases 8, 56, 34 and 23, while proecedures and medications are \[18,34,26,4\] and \[82,13,73,4\]\] respectively.

\[\[\[4,3,6,5\],\[34,2,35,3\],\[8,75,6,34\]\],\[\[8,56,34,23\],\[18,34,26,4\],\[82,13,73,4\]\]\]

To prepare datasets for MRSC, one needs to prepare the following files and put them into a file named as "data" in the project:

DIAGNOSES_ICD.csv

PROCEDURES_ICD.csv

PRESCRIPTIONS.csv (you can download these three files from [MIMIC](https://mimic.mit.edu))

ndc2atc_level4.csv

ndc2rxnorm_mapping.txt (you can find these two files in MRSC/data)

THen run the file DataProcessing.py

# Codes

DataProcessing: preapre datasets that are required to run MRSC

Networks: basic components of the model, including the encoder, decoders based on sum based coverage and GRU based coverage.

GRUCoverOptim.py: hyper-parameters tuning for GRU based coverage model

SumCoverOptim.py: hyper-parameters tuning for sum based coverage model

Optimization.py: basic components for hyper-parameters tuning

Training.py: model training

Evaluation.py: model evaluation

To carry out the recommendation, one could firstly run the codes in GRUCoverOptim.py and SumCoverOptim.py to get all hyper-parameters for the model. The tuning process is conducted based on bayesian optimization with [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize). These hyper-parameters could then be used to train and evaluate the recommendation model by runing Training.py and evaluation.py.
