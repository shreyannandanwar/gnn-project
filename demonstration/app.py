import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
import random

# -----------------------------
# Utility: Seed Control
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

# -----------------------------
# Task Definitions
# -----------------------------
TASKS = ["Tox21", "ClinTox", "BBBP", "BACE", "HIV"]

# -----------------------------
# Simple Mock 3D Encoder
# -----------------------------
class Mock3DEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return torch.relu(self.fc(x))

# -----------------------------
# Task Conditioned Model
# -----------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks=5, task_dim=16):
        super().__init__()
        self.encoder = Mock3DEncoder()
        self.task_embeddings = nn.Embedding(num_tasks, task_dim)
        self.heads = nn.ModuleList([nn.Linear(64 + task_dim, 1) for _ in range(num_tasks)])
        
    def forward(self, x, task_id):
        h = self.encoder(x)
        task_emb = self.task_embeddings(torch.tensor(task_id))
        task_emb = task_emb.unsqueeze(0).expand(h.size(0), -1)
        h = torch.cat([h, task_emb], dim=1)
        return self.heads[task_id](h)

# -----------------------------
# Fake Gradient Conflict Matrix
# -----------------------------
def generate_gradient_matrix(pcgrad=False):
    mat = np.random.uniform(-0.4, 0.8, (5, 5))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1)
    
    if pcgrad:
        mat = np.clip(mat, 0, 1)
    return mat

# -----------------------------
# Fake ROC Results
# -----------------------------
def generate_results(pcgrad=False):
    base = np.array([0.74, 0.71, 0.76, 0.73, 0.75])
    if pcgrad:
        base += np.array([0.02, 0.03, 0.01, 0.02, 0.02])
    return base

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Task-Conditioned Multi-Task 3D GNN Demo")

st.markdown("### Molecular Input")
smiles = st.text_input("Enter SMILES:", "CCO")

if st.button("Generate 3D Conformer"):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    st.success("3D Conformer Generated Successfully!")

st.markdown("---")

st.markdown("### Model Configuration")
use_pcgrad = st.checkbox("Enable PCGrad (Gradient Conflict Mitigation)")

model = MultiTaskModel()

task_choice = st.selectbox("Select Task:", TASKS)
task_id = TASKS.index(task_choice)

if st.button("Run Prediction"):
    dummy_input = torch.randn(1, 10)
    output = model(dummy_input, task_id)
    prob = torch.sigmoid(output).item()
    st.write(f"Predicted Probability ({task_choice}): {prob:.4f}")

st.markdown("---")

st.markdown("### Gradient Conflict Matrix")

matrix = generate_gradient_matrix(pcgrad=use_pcgrad)

fig, ax = plt.subplots()
cax = ax.matshow(matrix)
plt.xticks(range(5), TASKS, rotation=45)
plt.yticks(range(5), TASKS)
plt.colorbar(cax)
st.pyplot(fig)

st.markdown("---")

st.markdown("### ROC-AUC Benchmark Comparison")

results = generate_results(pcgrad=use_pcgrad)

fig2, ax2 = plt.subplots()
ax2.bar(TASKS, results)
ax2.set_ylim([0.6, 0.85])
ax2.set_ylabel("ROC-AUC")
st.pyplot(fig2)

st.success("Demo Completed Successfully!")
