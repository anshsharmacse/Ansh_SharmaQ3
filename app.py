#pip install streamlit - not required
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class VAE(torch.nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(28*28 + 10, 400)
        self.fc21 = torch.nn.Linear(400, latent_dim)
        self.fc22 = torch.nn.Linear(400, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim + 10, 400)
        self.fc4 = torch.nn.Linear(400, 28*28)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

def one_hot_label(digit, device):
    y = torch.zeros(1, 10).to(device)
    y[0, digit] = 1
    return y

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

# Web UI
st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate"):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20).to(device)
        y = one_hot_label(digit, device)
        sample = model.decode(z, y).cpu().detach().numpy().reshape(28, 28)
        axes[i].imshow(sample, cmap='gray')
        axes[i].axis("off")
    st.pyplot(fig)
