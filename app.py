import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha QP Optimizer", layout="wide")
st.title("📈 Mourabaha Portfolio Optimization Framework")

@st.cache_data
def get_optimized_portfolio():
    # 1. قراءة البيانات (تجاهل الأسطر التالفة)
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
    
    # 2. تنظيف البيانات (حسب التقرير: Winsorization & Interpolation)
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    
    # 3. حساب المدخلات للنموذج (Mean & Covariance Matrix)
    mu = numeric_df.mean().values
    sigma = numeric_df.cov().values
    
    # 4. الركن الأساسي: ضمان أن المصفوفة Σ "موجبة معرفة" (Positive Definite)
    # إضافة ضجيج صغير جداً (1e-6) للأقطار يمنع خطأ division by zero
    sigma = sigma + np.eye(len(mu)) * 1e-6
    
    return mu, sigma, numeric_df.columns

# تحميل البيانات
try:
    mu, sigma, sectors = get_optimized_portfolio()
    
    if st.button("🚀 Lancer l'optimisation QP"):
        n = len(mu)
        
        # دالة الهدف: min xᵀ Σ x (تقليل المخاطر/التباين)
        def objective(x):
            return np.dot(x.T, np.dot(sigma, x))
        
        # القيود: مجموع الأوزان = 1، والأوزان موجبة [0,1]
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # الحل باستخدام خوارزمية Sequential Least Squares Programming
        res = minimize(objective, [1/n]*n, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            st.success("Optimisation réussie !")
            res_df = pd.DataFrame({'Secteur': sectors, 'Poids Optimal (%)': (res.x * 100).round(2)})
            
            # العرض البياني
            st.subheader("Répartition Sectorielle")
            st.bar_chart(res_df.set_index('Secteur'))
            st.table(res_df)
        else:
            st.error("L'optimiseur n'a pas convergé.")

except Exception as e:
    st.error(f"Erreur système : {e}")
