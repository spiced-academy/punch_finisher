# 🎯 Kickstarter Campaign Success Prediction

In recent years, crowdfunding has emerged as a popular funding model for individuals and small companies to bring their ideas to life. Among the various platforms, **Kickstarter** stands out due to its **all-or-nothing** funding approach — a project only receives funds if it reaches its target goal.

The goal of this project is to:
- Analyze real Kickstarter data
- Identify key factors that influence project success
- Build a **predictive model** that estimates whether a project will succeed or fail.

This project is part of a 5-day final sprint at Spiced Academy, where we apply everything we’ve learned — from data cleaning and EDA to classification models and evaluation.

---

## 📁 Repository Structure

```
kickstarter-prediction/
│
├── data/                 # Contains raw and processed data
├── notebooks/            # EDA and modeling notebooks
├── src/                  # Reusable scripts for data handling and modeling
├── outputs/              # Visualizations and reports
├── requirements.txt      # Python dependencies
├── README.md             # Project overview
└── .gitignore            # Files to be ignored by Git
```

---

## ⚙️ Set up your Environment

### **`macOS`** type the following commands:

- For installing the virtual environment and the required packages, you can either follow these commands:

    ```bash
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    Or...

- Use the [Makefile](Makefile) and run `make setup`:

    ```bash
    make setup
    ```

    After that, activate your environment with:

    ```bash
    source .venv/bin/activate
    ```

---

### **`WindowsOS`** type the following commands:

- Install the virtual environment and the required packages using one of the following options:

#### For `PowerShell` CLI:
```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### For `Git-bash` CLI:
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:**  
> If you encounter an error with `pip install --upgrade pip`, try:
> ```bash
> python.exe -m pip install --upgrade pip
> ```

---

## 🚧 Work in Progress

This project is still under active development. Stay tuned for:
- Data Cleaning & EDA
- Feature Engineering
- Model Building & Evaluation
- Insights & Recommendations for creators
