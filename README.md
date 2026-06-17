# 🤖 CSV Insight Agents

**AI-Powered Multi-Agent System for Automated Data Analysis**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/csv-insight-agents.git
cd csv-insight-agents

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## ✨ Features

| Agent | Function |
|---|---|
| 🔍 **Data Quality** | Automatic quality scoring, completeness analysis, duplicate detection |
| 🧹 **Data Cleaning** | Handle missing values, remove duplicates, outlier treatment |
| 📊 **Statistics** | Descriptive stats, correlations, data profiling |
| 📈 **Visualization** | Interactive charts, heatmaps, distributions |
| 🤖 **Modeling** | Multiple ML algorithms, model comparison, performance metrics |
| 📋 **AI Report** | AI-generated insights and comprehensive analysis |

---

## 🎨 Theme

Premium dark theme with a custom `#1371A0` color palette:

- **Primary** — `#1371A0`
- **Secondary** — `#3188AD`
- **Accent** — `#77B4C7`

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit, Custom CSS |
| **Visualization** | Plotly |
| **ML / AI** | Scikit-learn, XGBoost, LightGBM |
| **LLM** | OpenRouter API (GPT-OSS-120B) |
| **Data** | Pandas, NumPy, SciPy |

---

## 📁 Project Structure

```
csv-insight-agents/
├── app.py              # Main application
├── pages_content/      # Page modules
├── utils/              # Utilities
└── requirements.txt    # Dependencies
```

---

## 🔑 API Key (Optional)

AI-powered features require an OpenRouter API key. You can provide it in one of two ways:

**Option 1 — secrets file** (`.streamlit/secrets.toml`):

```toml
OPENROUTER_API_KEY = "your-api-key"
```

**Option 2** — enter it directly in the sidebar at runtime.

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
scikit-learn>=1.2.0
scipy>=1.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
openpyxl>=3.1.0
requests>=2.28.0
imbalanced-learn>=0.10.0
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/amazing`
3. Commit your changes — `git commit -m 'Add feature'`
4. Push to the branch — `git push origin feature/amazing`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for data enthusiasts</p>
<p align="center">
  <a href="#">Report Bug</a> · <a href="#">Request Feature</a>
</p>
