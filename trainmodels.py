import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from VAE import VAE

#Load and etract the 2000 backwall a-scans
data = np.load(r'C:\Users\caleb\Downloads\000000003002031138\10MHz_conventional_0deg_aluminum_corrosion_p75in.npy')
tdata = data[:200,:-1]
data = data[200:2000,:-1]
print("Number of rows in training data: "+str(len(data)))
print("Number of columns in training data: "+str(len(data[0])))

#plot the first data point to ensure everything looks okay
ascan_num_to_plot = 0

#normalize the data
data = data/np.max(data)

#stats description of first reflection
firstref_amp = pd.DataFrame()
full_firstref = pd.DataFrame(pd.DataFrame(data).iloc[:,600:700])
for ascan in range(0,len(full_firstref)):
  ma = pd.DataFrame(full_firstref.iloc[ascan])
  ma = ma.max()
  firstref_amp = pd.concat([firstref_amp,ma])

#print all occurences of an abnormally large signal
largerows = full_firstref[full_firstref.gt(0.2).any(axis=1)].index

##uncomment HERE to plot all a-scans with abnormally large amplitude
# for value in largerows:
#     plt.figure()  # Create a new figure for each row
#     pd.DataFrame(data[value]).plot(kind='line', marker='o')
#     plt.title(f'Line Plot for Row {value}')
#     plt.xlabel('Samples')
#     plt.ylabel('Normalized Amplitude')
#     plt.grid(True)
#     plt.show()
##TO HERE

#Drop all rows with abnormally large peak value
data = pd.DataFrame(data).drop(largerows)
data = data.to_numpy()

# Hyperparameters
input_dim = data.shape[1]  # 1300
latent_dim = 100  # Latent space dimension
lr = 1e-4
batch_size = 64
epochs = 100

# Prepare data loader
dataset = TensorDataset(torch.Tensor(data))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
vae = VAE(input_dim, latent_dim)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

vae.apply(init_weights)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# Training loop
vae.train()

#dataframe to hold loss data
lossdf_50 = pd.DataFrame()

# Loss function (VAE Loss: Reconstruction loss + KL divergence)
def loss_function(recon_x, x, mean, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # print("Recon loss for this epoch is "+str(recon_loss))
    # print("Recon var1 (recon_x): "+str(recon_x))
    # print("Recon var2 (x): "+str(x))
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_div


for epoch in range(epochs):
    train_loss = 0
    for batch in dataloader:
        x = batch[0]
        # print("x is: "+str(x))
        # print("NaNs in x: ", torch.isnan(x).sum().item())  # Check for NaNs
        # print("Infs in x: ", torch.isinf(x).sum().item())  # Check for infinities
        optimizer.zero_grad()
        recon_x, mean, log_var = vae(x)
        # print("recon_x: "+str(recon_x))
        # print("mean: "+str(mean))
        # print("log_var: "+str(log_var))
        loss = loss_function(recon_x, x, mean, log_var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        train_loss += loss.item()
        optimizer.step()
    ltoadd = train_loss / len(dataloader.dataset)
    ltoadd = pd.DataFrame([ltoadd])
    lossdf_50 = pd.concat([lossdf_50,ltoadd])
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}')


# Define file path to save the model
filepath = r"C:\Users\caleb\Documents\5122\week8lab\pages_demo\backwall_vae_150e.pth"
# Save the model
torch.save(vae.state_dict(), filepath)
# Assuming your saved VAE model is at 'path/to/my_vae.h5'
model_state = torch.load(filepath)
vae_model = VAE(input_dim, latent_dim)  # 1300
vae_model.load_state_dict(model_state)
vae = vae_model
with torch.no_grad():
    z = torch.randn(1800, latent_dim)  # Generate 2000 new synthetic rows
    synthetic_data = vae.decode(z).numpy()
    print(synthetic_data)  # The generated rows
print()

lossdf_5 = lossdf_50.reset_index()
lossdf_5 = lossdf_5.drop('index',axis=1)
plt.plot(lossdf_5, linewidth=3.0)


plt.title("Loss (VAE MSE) | Learning Rate of 1e-5")
plt.xlabel("Epoch")
plt.ylabel("VAE MSE")

# Annotating the first and last points - lr5
first_value = float(lossdf_5.iloc[0])  # Convert to float
last_value = float(lossdf_5.iloc[-1])  # Convert to float
min_value = float(lossdf_5.iloc[int(len(lossdf_5)/2)])  # Convert to float

# Use format for numerical values
plt.text(0, first_value, f'{first_value:.2f}', ha='center', va='bottom',color="blue")  # First value at index 0
plt.text(int(len(lossdf_5)/2), min_value, f'{min_value:.2f}', ha='center', va='bottom',color="blue")  # Min loss value
plt.text(len(lossdf_5)-1, last_value, f'{last_value:.2f}', ha='center', va='bottom',color="blue")  # Last value
plt.show()
print()