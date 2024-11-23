# python3 -m venv venv
# source venv/bin/activate

import streamlit as st
from VAE import VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

st.set_page_config(layout="wide")


st.title("Please select which model you want to synthesize from, how much data you want to synthesize, and which A-scan to view in your new synthetic dataset:")
ascan_num = st.text_input("Enter Which A-scan You Want to View Here:", 1)
totscans = st.text_input("Enter The Number of A-scans You Want to Synthesize (Has to Be Greater Than 0):", 1)

def generate_synthetic_data():
  
  input_dim = 1300  # 1300
  latent_dim = 100  # Latent space dimension
  lr = 1e-5
  batch_size = 64
  epochs = 200
  filepath = r"C:\Users\caleb\Documents\5122\week8lab\pages_demo\backwall_vae_100e.pth"
  model_state = torch.load(filepath)
  vae_model = VAE(input_dim, latent_dim)  # 1300
  vae_model.load_state_dict(model_state)
  vae = vae_model
  with torch.no_grad():
    z = torch.randn(int(totscans), latent_dim)  # Generate 2000 new synthetic rows
    synthetic_data = vae.decode(z).numpy()
    print(synthetic_data)  # The generated rows
  synthetic_datanp = synthetic_data
  synthetic_data = pd.DataFrame(synthetic_data)
  x = synthetic_data.iloc[int(ascan_num)-1,:]
  micros = np.arange(1300)
  micros = micros/100
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.set_title("Synthetic A-scan #"+str(ascan_num))
  ax.set_ylabel("Normalized Amplitude")
  ax.set_xlabel("Microseconds")
  ax.plot(micros,x)
  # Display the plot in Streamlit
  st.pyplot(fig)


  data = np.load(r'C:\Users\caleb\Documents\5122\week8lab\pages_demo\testdata.npy')
  data = data/np.max(data)
  realmean = np.mean(abs(data), axis=0)
  synmean = np.mean(abs(synthetic_datanp), axis=0)
  side = pd.DataFrame(range(0,len(realmean)))*.01
  side = side.transpose()
  side = side.iloc[0,:]
  side1 = pd.DataFrame(range(0,len(synmean)))*.01
  side1 = side1.transpose()
  side1 = side1.iloc[0,:]
  fig1, ax1 = plt.subplots(figsize=(5, 5))
  ax1.plot(side, realmean)
  ax1.plot(side1, synmean)
  ax1.set_title("Mean of Real vs Synthetic Backwall A-scans")
  ax1.set_xlabel("Time (Microseconds)")
  ax1.set_ylabel("Normalized Amplitude")
  ax1.legend(["Real","Synthetic"])  

  realstd = np.std(abs(data), axis=0)
  synstd = np.std(abs(synthetic_datanp), axis=0)
  fig2, ax2 = plt.subplots(figsize=(5, 5))
  ax2.plot(side, realstd)
  ax2.plot(side1, synstd)
  ax2.set_title("Standard Deviation of Real vsSynthetic Backwall A-scans")
  ax2.set_xlabel("Time (Microseconds)")
  ax2.set_ylabel("Normalized Amplitude")
  ax2.legend(["Real","Synthetic"])

  firstref_amp = pd.DataFrame()
  full_firstref = pd.DataFrame(pd.DataFrame(data).iloc[:,600:700])
  for ascan in range(0,len(full_firstref)):
    ma = pd.DataFrame(full_firstref.iloc[ascan])
    ma = ma.max()
    firstref_amp = pd.concat([firstref_amp,ma])

  firstref_amp_syn = pd.DataFrame()
  full_firstref_syn = pd.DataFrame(pd.DataFrame(synthetic_datanp).iloc[:,600:700])
  for ascan in range(0,len(full_firstref_syn)):
    ma = pd.DataFrame(full_firstref_syn.iloc[ascan])
    ma = ma.max()
    firstref_amp_syn = pd.concat([firstref_amp_syn,ma])
  fig3, ax3 = plt.subplots(figsize=(5, 5))
  ax3.hist(firstref_amp,edgecolor='black', linewidth=1.2)
  ax3.set_title("Distribution of First Reflection Max Amplitudes of Real Data")
  ax3.set_xlabel("Amplitude")
  ax3.set_ylabel("Count")
  ax3.legend(["Real","Synthetic"])

  fig4, ax4 = plt.subplots(figsize=(5, 5))
  ax4.hist(firstref_amp_syn,edgecolor='black', linewidth=1.2)
  ax4.set_title("Distribution of First Reflection Max Amplitudes of Synthetic Data")
  ax4.set_xlabel("Amplitude")
  ax4.set_ylabel("Count")
  ax4.legend(["Real","Synthetic"])
  st.pyplot(fig1)
  st.pyplot(fig2)
  st.pyplot(fig3)
  st.pyplot(fig4)
  st.image("Figure_1.png")


option = st.selectbox(
    "Which Data Type Would You Like to Synthesize?",
    ("-","Backwall", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL", "PLACEHOLDER FOR ANOTHER MODEL"),
)
if str(option) == "Backwall":
  generate_synthetic_data()
# option = st.selectbox(
#     "How would you like to be contacted?",
#     ("Backwall", "Home phone", "Mobile phone"),
# )
# if st.button('Backwall'):
#   generate_synthetic_data()