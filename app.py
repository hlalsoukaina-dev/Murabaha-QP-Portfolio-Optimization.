import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.title("📈 Mourabaha Portfolio Optimization Framework")

@st.cache_data
def get_optimized_portfolio():
    # 1. تحميل البيانات
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    
    # 2. مصفاة التصفير: حذف القطاعات التي تباينها صفر تماماً (تمنع division by zero)
    cols_to_keep = numeric_df.columns[numeric_df.std() > 0]
    numeric_df = numeric_df[cols_to_keep]
    
    # 3. حساب المتوسط والمصفوفة
    mu = numeric_df.mean().values
    sigma = numeric_df.cov().values
    
    # 4. إضافة معامل استقرار رياضي (Regularization)
    sigma = sigma + np.eye(len(mu)) * 1e-6
    
    return mu, sigma, numeric_df.columns

try:
    mu, sigma, sectors = get_optimized_portfolio()
    
    if st.button("🚀 Lancer l'optimisation QP"):
        n = len(mu)
        
        def objective(x):
            return np.dot(x.T, np.dot(sigma, x))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        res = minimize(objective, [1/n]*n, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            st.success("Optimisation réussie !")
            res_df = pd.DataFrame({'Secteur': sectors, 'Poids Optimal (%)': (res.x * 100).round(2)})
            st.bar_chart(res_df.set_index('Secteur'))
            st.table(res_df)
        else:
            st.error("L'optimiseur n'a pas convergé. Vérifiez vos données.")

except Exception as e:
    st.error(f"Erreur fatale : {e}")
