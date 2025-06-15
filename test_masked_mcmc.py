from copy import deepcopy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rich import print

from catalax import Model
from catalax.dataset import Dataset
from catalax.dataset.measurement import Measurement
from catalax.mcmc import run_mcmc
from catalax.mcmc.priors import Uniform

# Create a simple exponential decay model
model = Model(name="exponential_decay")
model.add_species("Substrate, Enzyme")
model.add_ode("Substrate", "-k * Substrate * Enzyme")
model.add_ode("Enzyme", "-decay * Enzyme")

# Set up the model parameters with a truncated normal prior (ensures k > 0)
model.parameters["k"].prior = Uniform(low=1.0, high=100.0)
model.parameters["decay"].prior = Uniform(low=0.0, high=1)

# Create synthetic data with some NaN points
times = jnp.linspace(0, 1, 50)  # Shorter time range
true_k = 10.0
sub_0 = 1.0
enzy_0 = 0.5
true_decay = 0.4

sim_enzyme = enzy_0 * jnp.exp(-true_decay * times)
# Correct analytical solution for substrate
sim_substrate = sub_0 * jnp.exp(
    (true_k * enzy_0 / true_decay) * (jnp.exp(-true_decay * times) - 1)
)

# add some noise
noise = 0.01 * jax.random.normal(key=jax.random.key(0), shape=sim_substrate.shape)
noisy_substrate = sim_substrate + noise
noisy_enzyme = sim_enzyme + noise


# Create dataset
dataset = Dataset(species=["Substrate", "Enzyme"])
dataset.add_measurement(
    Measurement(
        initial_conditions={"Substrate": sub_0, "Enzyme": enzy_0},
        time=times,
        data={
            "Substrate": noisy_substrate,
            "Enzyme": noisy_enzyme,
        },
    )
)

masked_enzyme_dataset = deepcopy(dataset)
masked_enzyme_dataset.measurements[0].data["Enzyme"] = jnp.full_like(
    noisy_enzyme, jnp.nan
)

print(dataset)
print(masked_enzyme_dataset)

# Debug: Check data shape after processing
data_processed, times_processed, y0s_processed = dataset.to_jax_arrays(
    model.get_observable_species_order(),
    inits_to_array=True,
)

# Create mask - use OBSERVABLE species order to match data processing
mask = dataset.get_nan_mask(model.get_observable_species_order())


print(f"Mask shape: {mask.shape}")
print(f"Mask sum (number of True values): {mask.sum()}")
print(f"Data contains NaN: {jnp.isnan(data_processed).sum()} values")

# Verify mask and data shapes match
print(f"Mask shape matches data shape: {mask.shape == data_processed.shape}")

# Run MCMC with mask (skip no-mask version since data contains NaN)
print("\n=== Running MCMC with enzyme data ===")
full_data, _ = run_mcmc(
    model=model,
    dataset=dataset,
    yerrs=0.05,
    num_warmup=200,
    num_samples=500,
    verbose=1,
    dt0=0.001,  # Smaller time step for better numerical stability
    max_steps=2000,  # Increase max steps
)

print("\n=== Running MCMC with masked enzyme data ===")
masked_data, _ = run_mcmc(
    model=model,
    dataset=masked_enzyme_dataset,
    yerrs=0.05,
    num_warmup=200,
    num_samples=500,
    verbose=1,
    dt0=0.001,  # Smaller time step for better numerical stability
    max_steps=2000,  # Increase max steps
)

# Get posterior samples
samples_with_enzyme = full_data.get_samples()
samples_without_enzyme = masked_data.get_samples()


# Function to calculate predictions manually
def calculate_substrate(k_value, decay_value, times_array):
    """Calculate substrate concentration using correct analytical solution"""
    return sub_0 * jnp.exp(
        (k_value * enzy_0 / decay_value) * (jnp.exp(-decay_value * times_array) - 1)
    )


# Plot results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)

# Plot ground truth (simulated data)
plt.plot(
    times,
    sim_substrate,
    color="tab:blue",
    linewidth=2,
    label="Ground truth substrate",
    linestyle="--",
)
plt.plot(
    times,
    sim_enzyme,
    color="tab:orange",
    linewidth=2,
    label="Ground truth enzyme",
    linestyle="--",
)

# Plot noisy data (samples)
plt.scatter(
    times,
    noisy_substrate,
    color="tab:blue",
    alpha=0.3,
    label="Substrate data",
)
plt.scatter(
    times,
    noisy_enzyme,
    color="tab:orange",
    alpha=0.3,
    label="Enzyme data",
)

# Plot model fits from posterior (every 10th sample)
times_fine = jnp.linspace(0, 1, 100)
for i, (k, decay) in enumerate(
    zip(samples_with_enzyme["k"][::10], samples_with_enzyme["decay"][::10])
):  # Plot every 10th sample
    pred_substrate = calculate_substrate(k, decay, times_fine)
    line = plt.plot(
        times_fine, pred_substrate, color="tab:orange", alpha=0.3, linewidth=1
    )
    if i == 0:
        line[0].set_label("Fit")

for i, (k, decay) in enumerate(
    zip(samples_without_enzyme["k"][::10], samples_without_enzyme["decay"][::10])
):  # Plot every 10th sample
    pred_substrate = calculate_substrate(k, decay, times_fine)
    line = plt.plot(
        times_fine, pred_substrate, color="tab:blue", alpha=0.3, linewidth=1
    )
    if i == 0:
        line[0].set_label("Fit")

plt.title("MCMC Results: Model Comparison")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.grid(True, alpha=0.3)

# Plot parameter distributions
plt.subplot(1, 3, 2)
plt.hist(
    samples_with_enzyme["k"],
    bins=30,
    alpha=0.6,
    color="orange",
    label=f"With enzyme data (μ={samples_with_enzyme['k'].mean():.1f}±{samples_with_enzyme['k'].std():.1f})",
)
plt.hist(
    samples_without_enzyme["k"],
    bins=30,
    alpha=0.6,
    color="lightblue",
    label=f"Without enzyme data (μ={samples_without_enzyme['k'].mean():.1f}±{samples_without_enzyme['k'].std():.1f})",
)
plt.axvline(
    true_k, color="darkred", linestyle="--", linewidth=2, label=f"True k = {true_k}"
)
plt.xlabel("Rate constant k")
plt.ylabel("Frequency")
plt.title("Posterior Distribution of Rate Constant k")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot decay parameter distribution
plt.subplot(1, 3, 3)
plt.hist(
    samples_with_enzyme["decay"],
    bins=30,
    alpha=0.6,
    color="orange",
    label=f"With enzyme data (μ={samples_with_enzyme['decay'].mean():.1f}±{samples_with_enzyme['decay'].std():.1f})",
)
plt.hist(
    samples_without_enzyme["decay"],
    bins=30,
    alpha=0.6,
    color="lightblue",
    label=f"Without enzyme data (μ={samples_without_enzyme['decay'].mean():.1f}±{samples_without_enzyme['decay'].std():.1f})",
)
plt.axvline(
    true_decay,
    color="gray",
    linestyle="--",
    linewidth=2,
    label=f"True decay = {true_decay}",
)

plt.xlabel("Decay rate")
plt.ylabel("Frequency")
plt.title("Posterior Distribution of Decay Rate")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mcmc_enzyme_data_comparison.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'mcmc_enzyme_data_comparison.png'")
plt.close()

# Print parameter estimates
print("\nParameter estimates with enzyme data:")
print(
    f"k = {samples_with_enzyme['k'].mean():.3f} ± {samples_with_enzyme['k'].std():.3f}"
)
print(
    f"decay = {samples_with_enzyme['decay'].mean():.3f} ± {samples_with_enzyme['decay'].std():.3f}"
)

print("\nParameter estimates without enzyme data:")
print(
    f"k = {samples_without_enzyme['k'].mean():.3f} ± {samples_without_enzyme['k'].std():.3f}"
)
print(
    f"decay = {samples_without_enzyme['decay'].mean():.3f} ± {samples_without_enzyme['decay'].std():.3f}"
)

print("\nTrue values:")
print(f"k = {true_k:.3f}")
print(f"decay = {true_decay:.3f}")

print("\nSigma estimates:")
print(
    f"With enzyme data: σ = {samples_with_enzyme['sigma'].mean():.4f} ± {samples_with_enzyme['sigma'].std():.4f}"
)
print(
    f"Without enzyme data: σ = {samples_without_enzyme['sigma'].mean():.4f} ± {samples_without_enzyme['sigma'].std():.4f}"
)

# Check if enzyme data made a difference
k_diff = abs(samples_with_enzyme["k"].mean() - samples_without_enzyme["k"].mean())
decay_diff = abs(
    samples_with_enzyme["decay"].mean() - samples_without_enzyme["decay"].mean()
)
print("\nDifference in parameter estimates:")
print(f"k difference: {k_diff:.3f}")
print(f"decay difference: {decay_diff:.3f}")
