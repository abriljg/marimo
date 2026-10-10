import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Simulation of the Box-Muller Transform (from your Example)

    This notebook simulates the mathematical example provided in the image.

    **The Setup:**
    Let \(U \sim \text{Exponential}(1)\) and \(V \sim \text{Uniform}[0, 1]\), independent.
    We define:
    \[
    X = \sqrt{U} \cos(2\pi V), \quad Y = \sqrt{U} \sin(2\pi V)
    \]

    **The Theory:**
    As derived in the image, the joint density of \((X, Y)\) is:
    \[
    f_{X,Y}(x,y) = \frac{1}{\pi} e^{-x^2 - y^2}
    \]
    This factors into \(\frac{1}{\sqrt{\pi}}e^{-x^2} \cdot \frac{1}{\sqrt{\pi}}e^{-y^2}\), proving \(X\) and \(Y\) are independent, and each follows \(\mathcal{N}(0, 1/2)\).

    Use the slider below to see how the empirical simulation matches the theoretical distribution!
    """)
    return


@app.cell
def _(mo):
    # Interactive slider for the number of samples
    n_samples = mo.ui.slider(
        start=100, stop=100000, step=1000, value=10000, label="Number of Samples (N)"
    )
    n_samples
    return (n_samples,)


@app.cell
def _(n_samples, np):
    # 1. Generate the independent Uniform and Exponential variables
    # Using a fixed seed for reproducibility
    rng = np.random.default_rng(67)
    U = rng.exponential(scale=1.0, size=n_samples.value) # Exponential(1)
    V = rng.uniform(low=0.0, high=1.0, size=n_samples.value) # Uniform[0,1]

    # 2. Apply the Box-Muller transform (as per the example)
    # X = sqrt(U) * cos(2*pi*V)
    # Y = sqrt(U) * sin(2*pi*V)
    X = np.sqrt(U) * np.cos(2 * np.pi * V)
    Y = np.sqrt(U) * np.sin(2 * np.pi * V)
    return U, V, X, Y


@app.cell
def _(U, V, np, plt):
    # Visualize the Marginal Distributions of U=r^2 and V 
    fig_marg0, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 4))

    # Theoretical PDF for Exp(1): f(x) = exp(-x)
    u_vals = np.linspace(0, 4, 200)
    theo_pdf_u = np.exp(-u_vals)

    # Theoretical PDF for U(0, 1)
    v_vals = np.linspace(-1, 2, 200)
    theo_pdf_v = [1 if 0 <= p and p <= 1 else 0 for p in v_vals]

    # Plot Histogram of U
    ax_u.hist(U, bins=50, density=True, alpha=0.6, color='skyblue', label='Simulated U=r^2')
    ax_u.plot(u_vals, theo_pdf_u, 'r-', lw=2, label='Theoretical Exp(1)')
    ax_u.set_title("Marginal Distribution of U")
    ax_u.set_xlabel("U")
    ax_u.set_ylabel("Density")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)

    # Plot Histogram of V
    ax_v.hist(V, bins=50, density=True, alpha=0.6, color='lightgreen', label='Simulated V')
    ax_v.plot(v_vals, theo_pdf_v, 'r-', lw=2, label='Theoretical U(0, 1)')
    ax_v.set_title("Marginal Distribution of V")
    ax_v.set_xlabel("V")
    ax_v.set_ylabel("Density")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    fig_marg0
    return


@app.cell
def _(X, Y, np, plt):
    # 3. Visualize the 2D Joint Distribution
    fig_joint, ax_joint = plt.subplots(figsize=(6, 6))

    # Plot the scatter of (X, Y)
    ax_joint.scatter(X, Y, alpha=0.1, s=2, color='purple', label='Simulated Samples')

    # Overlay the theoretical density contours
    # The theoretical density is (1/pi) * exp(-x^2 - y^2)
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = (1 / np.pi) * np.exp(-X_grid**2 - Y_grid**2)

    # Plot contours at specific density levels (1 sigma, 2 sigma, etc.)
    ax_joint.contour(X_grid, Y_grid, Z_grid, levels=5, colors='black', linewidths=1.5)

    ax_joint.set_title("Joint Distribution of (X, Y)")
    ax_joint.set_xlabel("X")
    ax_joint.set_ylabel("Y")
    ax_joint.set_aspect('equal')
    ax_joint.grid(True, alpha=0.3)

    fig_joint
    return


@app.cell
def _(X, Y, np, plt):
    # 4. Visualize the Marginal Distributions and verify N(0, 1/2)
    fig_marg, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 4))

    # Theoretical PDF for N(0, 1/2): f(x) = (1/sqrt(pi)) * exp(-x^2)
    x_vals = np.linspace(-4, 4, 200)
    theo_pdf = (1 / np.sqrt(np.pi)) * np.exp(-x_vals**2)

    # Plot Histogram of X
    ax_x.hist(X, bins=50, density=True, alpha=0.6, color='skyblue', label='Simulated X')
    ax_x.plot(x_vals, theo_pdf, 'r-', lw=2, label='Theoretical N(0, 1/2)')
    ax_x.set_title("Marginal Distribution of X")
    ax_x.set_xlabel("X")
    ax_x.set_ylabel("Density")
    ax_x.legend()
    ax_x.grid(True, alpha=0.3)

    # Plot Histogram of Y
    ax_y.hist(Y, bins=50, density=True, alpha=0.6, color='lightgreen', label='Simulated Y')
    ax_y.plot(x_vals, theo_pdf, 'r-', lw=2, label='Theoretical N(0, 1/2)')
    ax_y.set_title("Marginal Distribution of Y")
    ax_y.set_xlabel("Y")
    ax_y.set_ylabel("Density")
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)

    fig_marg
    return


@app.cell
def _(X, Y, np):
    # 5. Statistical Verification of Independence and Parameters
    # Calculate the sample correlation
    corr_matrix = np.corrcoef(X, Y)
    sample_corr = corr_matrix[0, 1]

    # Calculate sample variance (should be close to 0.5)
    var_x = np.var(X)
    var_y = np.var(Y)
    return sample_corr, var_x, var_y


@app.cell(hide_code=True)
def _(mo, sample_corr, var_x, var_y):
    mo.md(rf"""
    ### Empirical Verification

    *   Sample Correlation \( \rho(X, Y) \): `{sample_corr:.4f}` *(Theoretical value is 0, indicating independence)*

    *   Sample Variance of X (\( \sigma_X^2 \)): `{var_x:.4f}` 
        *(Theoretical value is 0.5)*
    *   Sample Variance of Y (\( \sigma_Y^2 \)): `{var_y:.4f}` 
        *(Theoretical value is 0.5)*

    Because the correlation is practically zero and the variances match \(1/2\), this confirms 
    the theoretical result: X and Y are independent, and both follow \(\mathcal N(0, 1/2)\).
    """)
    return


if __name__ == "__main__":
    app.run()
