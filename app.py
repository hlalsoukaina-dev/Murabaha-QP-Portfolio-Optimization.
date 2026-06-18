import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha QP Optimizer", layout="wide")
st.title("📈 Mourabaha Portfolio Optimization Framework")

@st.cache_data
def process_data():
    # قراءة البيانات مع ضمان التعامل مع القيم المفقودة
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', engine='python')
    
    # تحويل البيانات إلى تنسيق رقمي نظيف (تجاهل الأعمدة غير الرقمية)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # ملء القيم الفارغة بـ 0 (كما في التقرير) ثم إضافة "ضجيج" تقني صغير جداً
    # لمنع الـ division by zero
    numeric_df = numeric_df.fillna(0) + 1e-9 
    
    mu = numeric_df.mean()
    sigma = numeric_df.cov()
    
    return mu, sigma

try:
    mu, sigma = process_data()
    st.success("Données chargées et traitées avec succès !")

    if st.button("Lancer Optimisation QP"):
        n = len(mu)
        
        # دالة الهدف: تباين المحفظة (Portfolio Variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(sigma.values, weights))
        
        # القيود: مجموع الأوزان = 1، والأوزان موجبة
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # استخدام خوارزمية تحسين أكثر استقراراً
        res = minimize(objective, [1/n]*n, method='SLSQP', 
                       bounds=bounds, constraints=constraints)
        
        if res.success:
            st.subheader("Allocation Optimale des Actifs")
            res_df = pd.DataFrame({'Secteur': mu.index, 'Poids (%)': (res.x * 100).round(2)})
            st.bar_chart(res_df.set_index('Secteur'))
            st.table(res_df)
        else:
            st.error("L'optimiseur n'a pas pu converger. Vérifiez la structure de votre matrice Σ.")

except Exception as e:
    st.error(f"Erreur technique (QP Framework): {e}")
