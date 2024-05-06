# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Figures for Illustration of concepts

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# %%
FIGUREFOLDER = Path('figures')
FIGUREFOLDER.mkdir(exist_ok=True, parents=True)


# %%
plt.rcParams.update({'xtick.labelsize': 'xx-large',
                     'ytick.labelsize': 'xx-large',
                     'axes.titlesize': 'xx-large',
                     'axes.labelsize': 'xx-large',
                     })
# {k:v for k,v in plt.rcParams.items() if 'tick' in k and 'size' in k}

# %% [markdown]
# ## Imputation by random draw from normal distribution
#
# - currently commenly used approach at NNF CPR for downstream experimentation task
# - see also illustrations in [Lazar 2016, Figure 1](https://pubs.acs.org/doi/10.1021/acs.jproteome.5b00981#fig1)

# %%
mu = 25.0
stddev = 1.0

x = np.linspace(mu - 3, mu + 3, num=101)

y_normal = scipy.stats.norm.pdf(x, loc=mu, scale=stddev)

mu_shifted = mu - (1.8 * stddev)
stddev_shifted = 0.3 * stddev
print(f"Downshifted: {mu_shifted = }, {stddev_shifted = }")
y_impute = scipy.stats.norm.pdf(x, loc=mu - (1.8 * stddev), scale=0.3 * stddev)

colors = plt.cm.viridis([0.25, 0.75])

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

for y, c in zip([y_normal, y_impute], colors):
    ax.plot(x, y, color=c,)
    ax.fill_between(x, y, color=c)
ax.set_xlabel('log2 intensity')
ax.set_ylabel('density')
ax.set_label("test")
ax.legend(["original", "down shifted"])
fig.tight_layout()

# %%

fig.savefig(FIGUREFOLDER / 'illustration_normal_imputation')
fig.savefig(FIGUREFOLDER / 'illustration_normal_imputation.pdf')
fig.savefig(FIGUREFOLDER / 'illustration_normal_imputation_highres', dpi=600)


# %%
plt.rcParams.update({'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.labelsize': 'large',
                     })
fig, ax = plt.subplots(1, 1, figsize=(3, 2))

for y, c in zip([y_normal], colors):
    ax.plot(x, y, color=c,)
    # ax.fill_between(x, y, color=c)
    ax.set_xlabel('log2 intensity')
    ax.set_ylabel('density')
    ax.set_label("test")
    # ax.legend(["original", "down shifted"])
fig.tight_layout()

fig.savefig(FIGUREFOLDER / 'illustration_normal')
fig.savefig(FIGUREFOLDER / 'illustration_normal.pdf')
fig.savefig(FIGUREFOLDER / 'illustration_normal_highres', dpi=600)


# %% [markdown]
# ## Log transformations and errors
#
# - what does log2 transformation mean for the error
#
# If the error is calculated in log2 space, the larger values have to be
# predicted with higher precision (in comparison to the original space)

# %%
def get_original_error_log2(x: float, error_log2: float):
    return 2 ** (np.log2(x) + error_log2) - x


print(
    f"{get_original_error_log2(1e9, 0.5) = :,.1f}",
    f"{get_original_error_log2(1e8, 0.5) = :,.1f}",
    sep='\n'
)


# %% [markdown]
# If we try to find the rel log2 error equalling the original error, this can be done by
# equating:
#
# $$ \exp(\ln(a)+e) - a = \exp(\ln(a)+e^*) - b $$
#
# Setting $a$, $e$ and $b$ we want to solve for $e^*$, which gives
#
# $$ e^* = \ln \left(\frac{\exp\big(\ln(a)+e\big) - a + b}{a}  \right)$$

# %%
def rel_error(measurment, log_error, other_measurment):
    numerator = 2 ** (np.log2(measurment) + log_error)
    numerator -= measurment
    numerator += other_measurment

    denominator = other_measurment
    return np.log2(numerator / denominator)


rel_error = rel_error(1.e9, 0.5, 1e8)
print(f"{rel_error = :.3f}")

# %%
print(
    f"0.500 rel to 1e9: {get_original_error_log2(1e9, 0.5) :,.1f}",
    f"{rel_error:.3f} rel to 1e8: {get_original_error_log2(1e8, rel_error) :,.1f}",
    sep='\n'
)

# %% [markdown]
# So the relative error of 0.5 for $10^9$ is five times larger for $10^8$ in the logspace,
# whereas the error in the original space is the same

# %% [markdown]
# ## Volcano plot

# %%
# Sample data for the volcano plot
np.random.seed(42)
fold_change = np.random.default_rng().normal(0, 1, 1000)
p_value = np.random.default_rng().uniform(0, 1, 1000)

# Volcano plot
# Assuming you have two arrays, fold_change and p_value, containing the fold change values and p-values respectively

# Set the significance threshold for p-value
significance_threshold = 0.05

# Set the fold change threshold
fold_change_threshold = 2

# Create a boolean mask for significant points
significant_mask = (p_value < significance_threshold)

# Create a boolean mask for points that meet the fold change threshold
fold_change_mask = (abs(fold_change) > fold_change_threshold)

# Combine the masks to get the final mask for significant points
final_mask = significant_mask & fold_change_mask

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# Plot the volcano plot
_ = ax.scatter(fold_change, -np.log10(p_value), c='gray', alpha=0.5, s=10)
_ = ax.scatter(fold_change[final_mask], -np.log10(p_value[final_mask]), c='red', alpha=0.7, s=20)

# Add labels and title
_ = ax.set_xlabel('Log2 fold change')
_ = ax.set_ylabel('-log10(p-value)')
_ = ax.set_title('Volcano Plot')

# Add significance threshold lines
_ = ax.axhline(-np.log10(significance_threshold), color='black', linestyle='--')
_ = ax.axvline(fold_change_threshold, color='black', linestyle='--')
_ = ax.axvline(-fold_change_threshold, color='black', linestyle='--')


# %%
fig.tight_layout()
fig.savefig(FIGUREFOLDER / 'illustration_volcano.png', dpi=300)
fig.savefig(FIGUREFOLDER / 'illustration_volcano.pdf')  
# %%
