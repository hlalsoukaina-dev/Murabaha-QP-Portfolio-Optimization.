# Mourabaha Portfolio Optimization: A Data-Driven Approach

## 📖 Project Overview
This research project focuses on the quantitative analysis and optimization of **Mourabaha** financing portfolios within the Moroccan banking sector. By leveraging historical data from 2019 to 2025, we aim to design a portfolio allocation model that maximizes returns while strictly adhering to the fundamental principles of **Islamic Finance** (Sharia-compliance), such as the prohibition of speculative practices (Gharar) and interest-based transactions (Riba).

## 👥 Research Team
This study is conducted by the 1st-year Master students of **Engineering in Participatory Finance and AI**:

* **Soukaina Hlal**
* **Asma Daaou**
* **Abdelouadoud Elkhalfi**
* **Mohammed Elgharb**

**Supervisor:** Dr. Asmae Faris

## 🎯 Objectives
- To evaluate the performance of different Mourabaha asset classes (Real Estate, Automotive, Equipment, and Commodities).
- To apply the **Modern Portfolio Theory (MPT)** framework adapted for the specific constraints of Participatory Finance.
- To demonstrate how Artificial Intelligence and statistical modeling can improve decision-making in Islamic banking.

## 🏗 Methodology
The project follows a rigorous quantitative workflow:
1.  **Data Preprocessing:** Cleaning and structuring raw financial datasets. We apply **Winsorization** (1%-99% quantiles) to handle extreme outliers and ensure model stability.
2.  **Portfolio Optimization:** Utilizing the **SLSQP (Sequential Least Squares Programming)** algorithm to solve the quadratic optimization problem.
3.  **Constraints Implementation:** The model enforces:
    * **Non-negativity constraint:** No short-selling allowed ($w_i \ge 0$).
    * **Budget constraint:** Total allocation must equal 100% ($\sum w_i = 1$).
4.  **Performance Metrics:** Calculation of expected returns, volatility, and Sharpe ratios to derive the efficient frontier.

## 🛠 Technical Stack
- **Language:** Python
- **Key Libraries:**
    - `Pandas` & `Numpy`: Data manipulation and numerical computation.
    - `Scipy.optimize`: Mathematical optimization algorithms.
    - `Matplotlib` & `Seaborn`: Statistical data visualization.

## 📂 Project Structure
- `Murabaha_QP_Framework_2019_2025_v2.ipynb`: Main Jupyter Notebook containing the full code, analysis, and visual outputs.
- `Murabaha_Quadratic_Ready_Data_2019_2025 (1).xlsx`: The processed dataset used for simulations.
- `requirements.txt`: List of dependencies for environment setup.

## 🚀 How to Run
1. Clone this repository: `git clone [repository_link]`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the analysis within the provided Jupyter Notebook.

---
*This repository serves as technical documentation accompanying the research work for the Master's degree in Engineering in Participatory Finance and AI.*
