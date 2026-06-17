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
@st.cache_data
def load_financial_data():
    try:
        # Loading files using the exact names from your GitHub
        mu = pd.read_csv('Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Expected_Returns_mu.csv', index_col=0)
        sigma = pd.read_csv('Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Covariance_Matrix_Sigma.csv', index_col=0)
        return mu, sigma
    except FileNotFoundError as e:
        st.error(f"File not found error: {e}. Please ensure the files are in the same folder as app.py")
        return None, None

mu, sigma = load_financial_data()

# --- Optimization Functions ---
def get_portfolio_performance(weights, mu, sigma):
    # Flatten mu to 1D array for dot product
    returns = np.dot(weights, mu.values.flatten())
    risk = np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
    return returns, risk

def objective(weights, mu, sigma):
    return get_portfolio_performance(weights, mu, sigma)[1]

# --- Main Logic ---
if mu is not None and sigma is not None:
    st.subheader("Asset Expected Returns")
    st.write(mu)

    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        # Weights must sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Weights must be between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Solving
        result = minimize(objective, initial_guess, args=(mu, sigma), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Display Results
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
