import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Page Configuration ---
st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")

# --- App Header ---
st.title("📈 Mourabaha Portfolio Optimization Dashboard")
st.markdown("""
This application utilizes **Quadratic Programming (SLSQP)** to determine the optimal asset allocation 
for a Sharia-compliant Mourabaha portfolio, based on historical financial data (2019-2025).
""")

# --- Sidebar Information ---
st.sidebar.header("Project Details")
st.sidebar.write("**Master Program:** Engineering in Participatory Finance and AI")
st.sidebar.write("**Team Members:** Soukaina, Asma, Abdelouadoud, Mohammed")
st.sidebar.write("**Supervisor:** Dr. Asmae Faris")

# --- Data Loading ---
# Note: Ensure these CSV files are in the same directory as this script
@st.cache_data
def load_financial_data():
    try:
        mu = pd.read_csv('Murabaha_Expected_Returns.csv', index_col=0)
        sigma = pd.read_csv('Murabaha_Covariance_Matrix.csv', index_col=0)
        return mu, sigma
    except FileNotFoundError:
        st.error("Error: Please ensure the CSV files are correctly named and uploaded.")
        return None, None

mu, sigma = load_financial_data()

# --- Portfolio Performance Function ---
def get_portfolio_performance(weights, mu, sigma):
    """Calculates portfolio returns and risk (volatility)."""
    returns = np.dot(weights, mu.values.flatten())
    risk = np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
    return returns, risk

# --- Optimization Objective ---
def objective(weights, mu, sigma):
    """Objective function: Minimize portfolio risk."""
    return get_portfolio_performance(weights, mu, sigma)[1]

# --- Main Logic ---
if mu is not None and sigma is not None:
    st.subheader("Asset Expected Returns")
    st.write(mu)

    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: No short-selling (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Solving the optimization problem
        result = minimize(objective, initial_guess, args=(mu, sigma), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Results Display
        st.subheader("Optimal Portfolio Allocation")
        results_df = pd.DataFrame({'Asset': mu.index, 'Weight': optimal_weights})
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Weight', y='Asset', data=results_df, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        st.write("Allocation Table (Percentage):")
        results_df['Weight (%)'] = (results_df['Weight'] * 100).round(2)
        st.write(results_df[['Asset', 'Weight (%)']])

# --- Footer ---
st.markdown("---")
st.caption("Research conducted under the framework of Participatory Finance and AI Engineering.")
