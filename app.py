import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha QP Optimizer", layout="wide")
st.title("📈 Murabaha Portfolio Optimization Framework")

@st.cache_data
def load_data():
    # المسارات الدقيقة بناءً على ملفاتك
    mu_file = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Expected_Returns_mu.csv'
    sigma_file = 'Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx - Covariance_Matrix_Sigma.csv'
    
    mu_df = pd.read_csv(mu_file, index_col=0)
    sigma_df = pd.read_csv(sigma_file, index_col=0)
    
    return mu_df, sigma_df

try:
    mu_df, sigma_df = load_data()
    
    # تحضير البيانات للمعادلة الرياضية
    returns = mu_df['mu_annuel_%'] / 100  # تحويل النسبة المئوية لكسر عشري
    cov_matrix = sigma_df.values
    assets = mu_df.index
    
    if st.button("🚀 Run Portfolio Optimization"):
        n = len(returns)
        
        # دالة الهدف: Minimizing Portfolio Variance (xT * Σ * x)
        def objective(x):
            return np.dot(x.T, np.dot(cov_matrix, x))
            
        # القيود: مجموع الأوزان = 1، والأوزان موجبة [0, 1]
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(n))
        
        result = minimize(objective, [1/n]*n, method='SLSQP', bounds=bnds, constraints=cons)
        
        # العرض
        st.subheader("Optimal Asset Allocation")
        res_df = pd.DataFrame({'Sector': assets, 'Optimal Weight (%)': (result.x * 100).round(2)})
        st.bar_chart(res_df.set_index('Sector'))
        st.table(res_df)
        
except Exception as e:
    st.error(f"Error: {e}")
