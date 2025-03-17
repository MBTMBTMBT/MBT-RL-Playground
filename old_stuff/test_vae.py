import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from vae import VAE

# Initialize latent dimension
latent_dim = 1
vae = VAE(
    in_channels=3,
    latent_dim=1,
    input_size=(60, 80),
    hidden_dims=[256, 512, 1024],
)
vae.load_state_dict(
    torch.load("../experiments/vae/checkpoints/vae_epoch_3.pth", map_location="cpu")
)

# Dynamically set latent_dim
st.sidebar.title("Settings")
latent_dim = st.sidebar.number_input(
    "Latent Dimension", min_value=1, max_value=512, value=latent_dim, step=1
)

# Initialize latent variables
latent_variables = np.zeros(latent_dim)

# Streamlit title
st.title("VAE Latent Space Explorer")
st.markdown(
    "Interactively adjust latent variables to explore the VAE's generative capabilities."
)

# Dynamically generate sliders
st.subheader("Adjust Latent Variables")
for i in range(latent_dim):
    latent_variables[i] = st.slider(f"Latent Variable {i}", -1.0, 1.0, 0.0, 0.01)

# Generate an image
latent_input = np.expand_dims(latent_variables, axis=0)  # Add batch dimension
latent_input_tensor = torch.tensor(latent_input, dtype=torch.float32)
generated_image = (
    vae.decode(latent_input_tensor).detach().numpy()[0]
)  # Assume decode outputs (C, H, W)

# Convert to Matplotlib-compatible image
generated_image = generated_image.transpose(1, 2, 0)  # Convert to (H, W, C)
generated_image = (generated_image - generated_image.min()) / (
    generated_image.max() - generated_image.min()
)  # Normalize

# Display the generated image
st.subheader("Generated Image")
st.image(
    generated_image,
    caption="Generated Image from Latent Variables",
    use_column_width=True,
)

# Display all latent variable values
if st.checkbox("Show All Latent Variables"):
    st.write(latent_variables)
