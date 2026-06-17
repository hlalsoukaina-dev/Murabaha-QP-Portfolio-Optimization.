import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha QP Optimizer", layout="wide")
st.title("📈 Murabaha Portfolio Optimization Framework")

@st.cache_data
def load_data():
    # البحث عن الملفات في المجلد الحالي بناءً على كلمات مفتاحية
    files = os.listdir('.')
    
    # البحث عن ملف mu و sigma
    mu_file = [f for f in files if 'Expected_Returns_mu' in f]
    sigma_file = [f for f in files if 'Covariance_Matrix_Sigma' in f]
    
    if not mu_file or not sigma_file:
        return None, f"Files not found! Directory contains: {files}"
    
    # تحميل البيانات
    mu_df = pd.read_csv(mu_file[0], index_col=0)
    sigma_df = pd.read_csv(sigma_file[0], index_col=0)
    
    return (mu_df, sigma_df), None

# --- تنفيذ الكود ---
data, error = load_data()

if error:
    st.error(error)
else:
    mu_df, sigma_df = data
    
    # تحضير البيانات
    returns = mu_df['mu_annuel_%'] / 100
    cov_matrix = sigma_df.values
    
    if st.button("🚀 Run Portfolio Optimization"):
        try:
            n = len(returns)
            def objective(x):
                return np.dot(x.T, np.dot(cov_matrix, x))
            
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bnds = tuple((0, 1) for _ in range(n))
            
            result = minimize(objective, [1/n]*n, method='SLSQP', bounds=bnds, constraints=cons)
            
            st.subheader("Optimal Asset Allocation")
            res_df = pd.DataFrame({'Sector': mu_df.index, 'Weight (%)': (result.x * 100).round(2)})
            st.bar_chart(res_df.set_index('Sector'))
            st.table(res_df)
        except Exception as e:
            st.error(f"Optimization Error: {e}")
