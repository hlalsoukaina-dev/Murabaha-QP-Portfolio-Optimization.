import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha Optimizer", layout="wide")
st.title("📈 Mourabaha Portfolio Optimization")

@st.cache_data
def load_data():
    # كايقلب على الملفات فالمجلد اللي فيه "Expected" و "Covariance"
    files = os.listdir('.')
    mu_file = [f for f in files if 'Expected_Returns' in f][0]
    sigma_file = [f for f in files if 'Covariance_Matrix' in f][0]
    
    # تحميل البيانات بأسماء ديناميكية
    mu = pd.read_csv(mu_file, index_col=0, encoding='latin-1')
    sigma = pd.read_csv(sigma_file, index_col=0, encoding='latin-1')
    
    return mu, sigma

try:
    mu, sigma = load_data()
    st.success("Données chargées avec succès !")
    
    if st.button("Run Optimization"):
        # تنظيف المصفوفة: ناخدو غير البيانات الرقمية
        sigma_matrix = sigma.select_dtypes(include=[np.number])
        # تنبيه: خصنا نتأكدو أن mu هي سيريز (Series)
        mu_values = mu['mu_annualise'] 
        
        num_assets = len(mu)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = [1./num_assets] * num_assets
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma_matrix.values, weights)))
        
        res = minimize(objective, initial_guess, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        st.subheader("Optimal Portfolio Allocation")
        results = pd.DataFrame({'Asset': mu.index, 'Weight': res.x})
        st.bar_chart(results.set_index('Asset'))
        st.write(results)

except Exception as e:
    st.error(f"Erreur de chargement: {e}")
