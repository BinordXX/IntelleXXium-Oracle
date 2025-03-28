# shap_analysis.py

import shap
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    print("Generating SHAP summary plot...")
    shap.plots.beeswarm(shap_values)
    plt.show()
