import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha Optimizer", layout="wide")
st.title("📈 Murabaha Portfolio Optimization")

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', 
                     on_bad_lines='skip', engine='python')
    
    # اختيار الأعمدة الرقمية فقط
    numeric_df = df.select_dtypes(include=[np.number])
    
    # تنظيف البيانات: استبدال NaN و Inf بـ 0
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # حساب المتوسط (mu) والمصفوفة (sigma)
    mu = numeric_df.mean()
    sigma = numeric_df.cov()
    
    # الحماية من القسمة على صفر: إضافة قيمة صغيرة جداً للأقطار
    # هذا يضمن أن المصفوفة دائماً قابلة للعكس (Invertible)
    epsilon = 1e-8
    sigma = sigma + np.eye(len(sigma)) * epsilon
    
    return mu, sigma

try:
    mu, sigma = load_and_process_data()
    st.success("Data processed successfully!")
    
    if st.button("Run Portfolio Optimization"):
        n = len(mu)
        
        # القيود: مجموع الأوزان = 1، والأوزان بين 0 و 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        def objective(weights):
            # التأكد من عدم وجود قيم فارغة في weights
            weights = np.array(weights)
            return np.sqrt(np.dot(weights.T, np.dot(sigma.values, weights)))
            
        # إعطاء نقطة بداية متساوية للجميع
        initial_guess = [1./n] * n
        
        res = minimize(objective, initial_guess, method='SLSQP', 
                       bounds=bounds, constraints=constraints)
        
        st.subheader("Optimal Portfolio Allocation")
        # عرض النتائج
        results = pd.DataFrame({'Asset': mu.index, 'Weight (%)': (res.x * 100).round(2)})
        st.bar_chart(results.set_index('Asset'))
        st.write(results)

except Exception as e:
    st.error(f"Error: {e}. Ensure 'mu.csv' has numeric return data.")
