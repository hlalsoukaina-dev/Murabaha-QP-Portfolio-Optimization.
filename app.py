import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Page Configuration ---
st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")

st.title("📈 Mourabaha Portfolio Optimization Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Using 'latin-1' encoding to handle special characters that caused the UnicodeDecodeError
    mu = pd.read_csv('mu.csv', index_col=0, encoding='latin-1')
    sigma = pd.read_csv('sigma.csv', index_col=0, encoding='latin-1')
    return mu, sigma

# --- Main Logic ---
try:
    mu, sigma = load_data()
    
    st.sidebar.header("Project Details")
    st.sidebar.write("**Master Program:** Engineering in Participatory Finance and AI")
    st.sidebar.write("**Team Members:** Soukaina, Asma, Abdelouadoud, Mohammed")
    st.sidebar.write("**Supervisor:** Dr. Asmae Faris")

    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: 0 to 1 (No short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Optimization: Minimize Portfolio Risk
        def objective(weights):
            # Portfolio Volatility = sqrt(w.T * Sigma * w)
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # Display Results
        st.subheader("Optimal Portfolio Allocation")
        results = pd.DataFrame({'Asset': mu.index, 'Weight': result.x})
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Weight', y='Asset', data=results, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        # Table
        results['Weight (%)'] = (results['Weight'] * 100).round(2)
        st.write(results[['Asset', 'Weight (%)']])

except Exception as e:
    st.error("Error: Could not load data.")
    st.write("Ensure 'mu.csv' and 'sigma.csv' are in the main folder and are valid CSV files.")
    st.write("Technical Error Details:", e)
