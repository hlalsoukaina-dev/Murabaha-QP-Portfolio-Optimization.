import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha Optimizer", layout="wide")
st.title("📈 Murabaha Portfolio Optimization")

@st.cache_data
def load_and_process_data():
    # قراءة الملف مع تنظيف الأسطر الخاطئة
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', 
                     on_bad_lines='skip', engine='python')
    
    # اختيار البيانات الرقمية فقط
    numeric_df = df.select_dtypes(include=[np.number])
    
    # تنظيف البيانات: استبدال القيم الفارغة (NaN) أو اللانهائية بـ 0
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # حساب المتوسط (mu) والمصفوفة (sigma)
    mu = numeric_df.mean()
    sigma = numeric_df.cov()
    
    # حماية إضافية: إذا كانت مصفوفة التباين بها أصفار، نضع قيمة صغيرة جداً لتجنب القسمة على صفر
    sigma = sigma + np.eye(len(sigma)) * 1e-6
    
    return mu, sigma

try:
    mu, sigma = load_and_process_data()
    st.success("Data processed successfully!")
    
    if st.button("Run Portfolio Optimization"):
        n = len(mu)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
            
        # استخدام قيمة أولية متساوية
        res = minimize(objective, [1./n]*n, method='SLSQP', bounds=bounds, constraints=constraints)
        
        st.subheader("Optimal Portfolio Allocation")
        results = pd.DataFrame({'Asset': mu.index, 'Weight (%)': (res.x * 100).round(2)})
        st.bar_chart(results.set_index('Asset'))
        st.write(results)

except Exception as e:
    st.error(f"Error: {e}. Check if 'mu.csv' contains valid numeric columns.")
