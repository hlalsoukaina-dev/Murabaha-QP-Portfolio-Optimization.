import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Mourabaha QP Optimizer", layout="wide")
st.title("📈 Murabaha Portfolio Optimization Framework")

@st.cache_data
def process_data():
    # استخدام 'python' engine كونه الأفضل في التعامل مع الأخطاء في الملفات
    # on_bad_lines='skip' ستقوم بتخطي السطر 23 وأي سطر آخر تالف
    df = pd.read_csv('mu.csv', index_col=0, encoding='latin-1', 
                     on_bad_lines='skip', engine='python')
    
    # تحويل البيانات إلى أرقام
    numeric_df = df.select_dtypes(include=[np.number])
    
    # ملء القيم الفارغة ومعالجة الأصفار الهيكلية (كما في تقريرك)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # إضافة قيمة صغيرة جداً لمنع أي division by zero مستقبلي
    numeric_df = numeric_df + 1e-9
    
    mu = numeric_df.mean()
    sigma = numeric_df.cov()
    
    return mu, sigma

try:
    mu, sigma = process_data()
    st.success("Données traitées : Le framework QP est prêt.")

    if st.button("Lancer Optimisation"):
        n = len(mu)
        
        # دالة الهدف: تباين المحفظة
        def objective(weights):
            return np.dot(weights.T, np.dot(sigma.values, weights))
        
        # القيود: Σx = 1, x >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        res = minimize(objective, [1/n]*n, method='SLSQP', 
                       bounds=bounds, constraints=constraints)
        
        if res.success:
            st.subheader("Allocation Optimale")
            res_df = pd.DataFrame({'Secteur': mu.index, 'Poids (%)': (res.x * 100).round(2)})
            st.bar_chart(res_df.set_index('Secteur'))
            st.table(res_df)
        else:
            st.warning("Optimisation impossible : vérifiez les données.")

except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    
