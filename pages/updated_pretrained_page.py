import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


# Function to load and normalize data
def load_and_normalize_data(df, start_row, end_row):
    data = df[start_row:end_row + 1, :-1]  # Exclude the last column
    data = data / np.max(data)  # Normalize data
    return data

# Function to split data into train and test sets
def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    print(f"Data split into {train_size} training and {len(test_data)} testing samples.")
    return train_data, test_data

# Function to plot histogram (pre-cleaning)
def plot_histogram_pre_cleaning(data, column_range, label):
    focus_columns = data[:, column_range[0]:column_range[1]]
    max_amplitudes = pd.DataFrame(focus_columns).max(axis=1)
    Q1, Q3 = max_amplitudes.quantile(0.25), max_amplitudes.quantile(0.75)
    upper_threshold = Q3 + 1.5 * (Q3 - Q1)

    plt.hist(max_amplitudes, bins=50, edgecolor='black', linewidth=1.2)
    plt.axvline(upper_threshold, color='red', linestyle='dashed', linewidth=1.5, label='Upper Threshold')
    plt.title(f"Histogram of Maximum Amplitudes (Pre-Cleaning, {label})")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    return upper_threshold
    

# Function to remove outliers 
def remove_outliers(data, column_range, multiplier=1.5):
    focus_columns = data[:, column_range[0]:column_range[1]]
    max_amplitudes = pd.DataFrame(focus_columns).max(axis=1)
    Q1, Q3 = max_amplitudes.quantile(0.25), max_amplitudes.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - multiplier * IQR, Q3 + multiplier * IQR

    outlier_indices = max_amplitudes[(max_amplitudes < lower_bound) | (max_amplitudes > upper_bound)].index
    data_cleaned = np.delete(data, outlier_indices, axis=0)
    return data_cleaned, outlier_indices


# Function to plot pre-cleaning metrics
def plot_pre_cleaning_metrics(data, label):
    real_mean, real_std = np.mean(data, axis=0), np.std(data, axis=0)
    plt.plot(real_mean, label="Mean (Pre-Cleaning)", linewidth=2)
    plt.title(f"Mean of Real Data (Pre-Cleaning, {label})")
    plt.xlabel("Time (Microseconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    plt.plot(real_std, label="Std Dev (Pre-Cleaning)", linewidth=2)
    plt.title(f"Standard Deviation of Real Data (Pre-Cleaning, {label})")
    plt.xlabel("Time (Microseconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

# Function to plot post-cleaning metrics
def plot_post_cleaning_metrics(data, label):
    real_mean, real_std = np.mean(data, axis=0), np.std(data, axis=0)
    plt.plot(real_mean, label="Mean (Post-Cleaning)", linewidth=2)
    plt.title(f"Mean of Real Data (Post-Cleaning, {label})")
    plt.xlabel("Time (Microseconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    plt.plot(real_std, label="Std Dev (Post-Cleaning)", linewidth=2)
    plt.title(f"Standard Deviation of Real Data (Post-Cleaning, {label})")
    plt.xlabel("Time (Microseconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    
# Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# Gradient Penalty Calculation
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# WGAN-GP Training with Loss Tracking, Early Stopping, and Patience
def train_wgan_gp_with_loss_tracking(data, latent_dim=100, batch_size=64, epochs=200, lambda_gp=10, n_critic=5, synthetic_samples=1000, patience=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(TensorDataset(torch.Tensor(data)), batch_size=batch_size, shuffle=True)
    
    input_dim = data.shape[1]
    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    train_losses, test_losses = [], []
    min_test_loss, patience_counter = float('inf'), 0

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        train_loss = 0

        for real_data_batch in train_loader:
            real_data = real_data_batch[0].to(device).requires_grad_(True)

            for _ in range(n_critic):
                optimizer_D.zero_grad()
                noise = torch.randn(real_data.size(0), latent_dim).to(device)
                fake_data = generator(noise).detach()
                real_validity = discriminator(real_data)
                fake_validity = discriminator(fake_data)
                gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()
            noise = torch.randn(real_data.size(0), latent_dim).to(device)
            fake_data = generator(noise)
            g_loss = -torch.mean(discriminator(fake_data))
            g_loss.backward()
            optimizer_G.step()

            train_loss += d_loss.item() + g_loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Check test loss for early stopping
        with torch.no_grad():
            test_loss = sum(d_loss.item() + g_loss.item() for real_data_batch in train_loader) / len(train_loader)
            test_losses.append(test_loss)

        # Early stopping check
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                st.write("Early stopping triggered")
                break
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Train-Test Loss Curve (WGAN-GP)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    generator.eval()
    with torch.no_grad():
        noise = torch.randn(synthetic_samples, latent_dim).to(device)
        synthetic_data = generator(noise).cpu().numpy()

    return synthetic_data, generator, discriminator

# Function to compare real and synthetic data
def compare_bin_real_and_synthetic_gan(real_data, synthetic_data, label=""):
    """
    Compare real (cleaned) and synthetic data for a specific bin.

    Parameters:
        real_data (numpy.ndarray): Cleaned real data.
        synthetic_data (numpy.ndarray): Generated synthetic data.
        label (str): Label for the bin (e.g., "0% Backwall").
    """
    # Calculate metrics for real data
    real_mean = np.mean(real_data, axis=0)
    real_std = np.std(real_data, axis=0)

    # Calculate metrics for synthetic data
    synthetic_mean = np.mean(synthetic_data, axis=0)
    synthetic_std = np.std(synthetic_data, axis=0)

    # Plot mean comparison
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(real_mean, label=f"Real Data Mean {label}", linewidth=2)
    ax1.plot(synthetic_mean, label=f"Synthetic Data Mean {label}", linestyle="--", linewidth=2)
    ax1.set_title(f"Mean Comparison: Real vs Synthetic {label} (GAN)")
    ax1.set_xlabel("Time (Microseconds)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend()
    st.pyplot(fig1)

    # Plot standard deviation comparison
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(real_std, label=f"Real Data Std Dev {label}", linewidth=2)
    ax2.plot(synthetic_std, label=f"Synthetic Data Std Dev {label}", linestyle="--", linewidth=2)
    ax2.set_title(f"Standard Deviation Comparison: Real vs Synthetic {label} (GAN)")
    ax2.set_xlabel("Time (Microseconds)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.legend()
    st.pyplot(fig2)

# Streamlit UI 
st.title('Generate synthetic UT data with GAN')

bins = [
    {"label": "Bin-1: 0% Backwall", "start": 0, "end": 1999, "samples": 2000},
    {"label": "Bin-2: 67.64% to 66.02%", "start": 2000, "end": 2999, "samples": 1000},
    {"label": "Bin-3: 66.02% to 64.40%", "start": 3000, "end": 3999, "samples": 1000},
    {"label": "Bin-4: 64.4% to 62.78%", "start": 4000, "end": 4733, "samples": 734},
    {"label": "Bin-5: 62.78% to 61.17%", "start": 4734, "end": 5220, "samples": 487},
    {"label": "Bin-6: 61.17% to 59.55%", "start": 5221, "end": 5618, "samples": 398},
    {"label": "Bin-7: 59.55% to 57.93%", "start": 5619, "end": 5829, "samples": 211},
    {"label": "Bin-8: 57.93% to 56.31%", "start": 5830, "end": 5949, "samples": 120},
    {"label": "Bin-9: 56.31% to 54.69%", "start": 5950, "end": 6949, "samples": 1000},
    {"label": "Bin-10: 54.69% to 53.07%", "start": 6950, "end": 7949, "samples": 1000},
    {"label": "Bin-11: 53.07% to 51.46%", "start": 7950, "end": 8671, "samples": 722},
    {"label": "Bin-12: 51.46% to 49.84%", "start": 8672, "end": 9332, "samples": 661},
    {"label": "Bin-13: 49.84% to 48.22%", "start": 9333, "end": 10332, "samples": 1000},
    {"label": "Bin-14: 48.22% to 46.60%", "start": 10333, "end": 11148, "samples": 816},
    {"label": "Bin-15: 46.6% to 44.98%", "start": 11149, "end": 11918, "samples": 770},
    {"label": "Bin-16: 44.98% to 43.37%", "start": 11919, "end": 12493, "samples": 575},
    {"label": "Bin-17: 43.37% to 41.75%", "start": 12494, "end": 13147, "samples": 654},
    {"label": "Bin-18: 41.75% to 40.13%", "start": 13148, "end": 13966, "samples": 819},
    {"label": "Bin-19: 40.13% to 38.51%", "start": 13967, "end": 14666, "samples": 700},
    {"label": "Bin-20: 38.51% to 36.89%", "start": 14667, "end": 15313, "samples": 647},
    {"label": "Bin-21: 36.89% to 35.28%", "start": 15314, "end": 15952, "samples": 639},
    {"label": "Bin-22: 35.28% to 33.66%", "start": 15953, "end": 16573, "samples": 621},
    {"label": "Bin-23: 33.66% to 32.04%", "start": 16574, "end": 17099, "samples": 526},
    {"label": "Bin-24: 32.04% to 30.42%", "start": 17100, "end": 17561, "samples": 462},
    {"label": "Bin-25: 30.42% to 28.80%", "start": 17562, "end": 17895, "samples": 334},
    {"label": "Bin-26: 28.80% to 27.18%", "start": 17896, "end": 18137, "samples": 242},
    {"label": "Bin-27: 27.18% to 25.57%", "start": 18138, "end": 18339, "samples": 202},
    {"label": "Bin-28: 25.57% to 23.95%", "start": 18340, "end": 18482, "samples": 143},
    {"label": "Bin-29: 23.95% to 22.33%", "start": 18483, "end": 18669, "samples": 187},
    {"label": "Bin-30: 22.33% to 20.71%", "start": 18670, "end": 18894, "samples": 225},
    {"label": "Bin-31: 20.71% to 19.09%", "start": 18895, "end": 19206, "samples": 312},
    {"label": "Bin-32: 19.09% to 17.48%", "start": 19207, "end": 19515, "samples": 309},
    {"label": "Bin-33: 17.48% to 15.86%", "start": 19516, "end": 19813, "samples": 298},
    {"label": "Bin-34: 15.86% to 14.24%", "start": 19814, "end": 20256, "samples": 443},
    {"label": "Bin-35: 14.24% to 12.62%", "start": 20257, "end": 21036, "samples": 780},
    {"label": "Bin-36: 12.62% to 11.00%", "start": 21037, "end": 21888, "samples": 852},
    {"label": "Bin-37: 11.00% to 9.39%", "start": 21889, "end": 22467, "samples": 579},
    {"label": "Bin-38: 9.39% to 7.77%", "start": 22468, "end": 22973, "samples": 506},
    {"label": "Bin-39: 7.77% to 6.15%", "start": 22974, "end": 23621, "samples": 648},
    {"label": "Bin-40: 6.15% to 4.53%", "start": 23622, "end": 24508, "samples": 887},
    {"label": "Bin-41: 4.53% to 2.91%", "start": 24509, "end": 25508, "samples": 1000},
    {"label": "Bin-42: 2.91% to 1.29%", "start": 25509, "end": 25891, "samples": 383}
]


selected_bin = st.selectbox("Select a corrosion level (bin)", range(len(bins)), format_func=lambda x: bins[x]["label"])

# Load the data for the selected bin
bin_info = bins[selected_bin]
df = np.load('10MHz_conventional_0deg_aluminum_corrosion_p75in.npy')


try:
    # Step 1: Load and normalize data
    data_bin = load_and_normalize_data(df, bin_info["start"], bin_info["end"])

    # Step 2: Plot pre-cleaning metrics
    plot_pre_cleaning_metrics(data_bin, label=bin_info["label"])

    # Step 3: Plot histogram of max amplitudes (pre-cleaning)
    upper_threshold = plot_histogram_pre_cleaning(data_bin, column_range=(600, 700), label=bin_info["label"])

    # Step 4: Remove outliers dynamically
    data_cleaned_bin, _ = remove_outliers(data_bin, column_range=(600, 700))

    # Step 5: Plot post-cleaning metrics
    plot_post_cleaning_metrics(data_cleaned_bin, label=bin_info["label"])

    # Step 6: Train WGAN-GP with loss tracking and early stopping, and generate synthetic data
    synthetic_data_bin, generator_bin, discriminator_bin = train_wgan_gp_with_loss_tracking(
        data_cleaned_bin,
        latent_dim=100,
        batch_size=64,
        epochs=200,
        lambda_gp=10,
        n_critic=5,
        synthetic_samples=bin_info["samples"],
        patience=12  # Early stopping patience
    )

    # Step 7: Compare real and synthetic data
    compare_bin_real_and_synthetic_gan(data_cleaned_bin, synthetic_data_bin, label=bin_info["label"])

except Exception as e:
    st.error(f"Error: {e}")

# Plot and display synthetic data
st.write(f"Generated synthetic data for {bin_info['label']}")

# Display synthetic data as a table
synthetic_df = pd.DataFrame(synthetic_data_bin)  
st.write("Synthetic Data Sample:", synthetic_df.head())  
