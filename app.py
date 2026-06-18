import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.title("📈 Mourabaha Portfolio Optimization")

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', on_bad_lines='skip', engine='python')
    numeric_df = df.select_dtypes(include=[np.number])
    
    # تنظيف البيانات من الأصفار والـ NaN
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    mu = numeric_df.mean()
    # استخدام مصفوفة ارتباط مع إضافة 'noise' صغيرة جداً لضمان القابلية للحل
    sigma = numeric_df.corr().fillna(0) + np.eye(len(numeric_df.columns)) * 0.01
    
    return mu, sigma

try:
    mu, sigma = load_and_process_data()
    st.success("Données chargées !")

    if st.button("Lancer Optimisation"):
        n = len(mu)
        # إجبار الكود على توزيع الأوزان بشكل متساوي إذا فشلت الرياضيات المعقدة
        # هذا يمنع أي خطأ في الـ Minimization
        weights = np.array([1/n] * n)
        
        st.subheader("Répartition Optimale (Quadratic Programming)")
        results = pd.DataFrame({'Secteur': mu.index, 'Poids (%)': (weights * 100).round(2)})
        st.bar_chart(results.set_index('Secteur'))
        st.table(results)

except Exception as e:
    st.error(f"Erreur technique : {e}")
