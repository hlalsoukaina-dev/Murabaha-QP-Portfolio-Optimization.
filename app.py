import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha Portfolio Optimizer", layout="wide")
st.title("📈 Mourabaha Portfolio Optimization Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    # Check if files exist in the current directory
    current_files = os.listdir('.')
    
    if 'mu.csv' not in current_files or 'sigma.csv' not in current_files:
        return None, f"Files not found. Files currently in directory: {current_files}"
    
    try:
        mu = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
        sigma = pd.read_csv('sigma.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
        return (mu, sigma), None
    except Exception as e:
        return None, f"Error reading CSV files: {e}"

# --- Main Logic ---
result_data, error = load_data()

if error:
    st.error(error)
else:
    mu, sigma = result_data
    st.success("Data loaded successfully!")
    
    if st.button("Run Portfolio Optimization"):
        num_assets = len(mu)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: 0 to 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        # Objective: Minimize Risk
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
        
        res = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        # Results
        st.subheader("Optimal Portfolio Allocation")
        results = pd.DataFrame({'Asset': mu.index, 'Weight': res.x})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Weight', y='Asset', data=results, palette="viridis", ax=ax)
        st.pyplot(fig)
        
        results['Weight (%)'] = (results['Weight'] * 100).round(2)
        st.write(results[['Asset', 'Weight (%)']])
