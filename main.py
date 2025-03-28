# main.py

from models.train_model import train_sales_model
from explainability.shap_analysis import run_shap_analysis

def main():
    print("🔮 Training the Oracle...")
    model, X_test = train_sales_model("data/sample_sales.csv")
    
    print("🧠 Explaining the Oracle's mind...")
    run_shap_analysis(model, X_test)

if __name__ == "__main__":
    main()
