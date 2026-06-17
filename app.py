import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# --- App Configuration ---
st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")

# --- Title and Header ---
st.title("📈 Mourabaha Portfolio Optimization Framework")
st.markdown("""
### Participatory Finance & AI Research Project
This application implements the **Modern Portfolio Theory (MPT)** adapted for **Sharia-compliant** Mourabaha financing assets.
""")

# --- Sidebar ---
st.sidebar.header("Project Information")
st.sidebar.write("**Team:** Soukaina, Asma, Abdelouadoud, Mohammed")
st.sidebar.write("**Supervisor:** Dr. Asmae Faris")

# --- Load Data ---
@st.cache_data
def load_data():
    # Make sure your file is in the same folder
    df = pd.read_excel('Murabaha_Quadratic_Ready_Data_2019_2025.xlsx')
    return df

try:
    data = load_data()
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")

# --- Portfolio Optimization Logic ---
def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.sum(returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return port_return, port_volatility

# Optimization constraint: weights sum to 1
def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.0):
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

# --- Main Interface ---
if st.checkbox("Show Raw Data"):
    st.write(data.head())

if st.button("Optimize Portfolio"):
    # Assuming the data has asset columns
    assets = data.columns[1:] 
    returns = data[assets].pct_change().dropna()
    cov_matrix = returns.cov()
    num_assets = len(assets)
    
    # Constraints & Bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimization
    result = minimize(negative_sharpe, initial_guess, args=(returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # Visualization
    st.subheader("Optimal Allocation Results")
    fig, ax = plt.subplots()
    sns.barplot(x=assets, y=optimal_weights, ax=ax, palette="viridis")
    st.pyplot(fig)
    
    st.write("Optimal Weights:", dict(zip(assets, np.round(optimal_weights, 4))))

# --- Footer ---
st.markdown("---")
st.write("This tool is part of the Master's research in Engineering in Participatory Finance and AI.")
