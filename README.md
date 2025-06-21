
# HR Attrition Risk & Upskilling Portal

A modern Streamlit solution for HR teams to analyze employee attrition risk, identify drivers, and generate personalized upskilling plans using machine learning and generative AI.

---

## 🚀 Live Demo

[**App Link**](https://attrisense-pro-genai-powered-hr-attrition-management.streamlit.app)  
*Replace this with your Streamlit Cloud URL after deployment.*

---

## 📝 Overview

**HR Attrition Risk & Upskilling Portal** provides:
- Interactive dashboard to explore attrition rates, drivers, and department/job-role breakdowns
- ML-powered risk prediction for each employee (XGBoost)
- Survival analysis for tenure prediction (Cox Proportional Hazards)
- AI Q&A and insights on attrition, cohorts, and individual employees (Gemini LLM)
- Automated, real-course-based upskilling plans for selected employees
- Modern UI with custom dark theme and right-aligned brand icon/banner

---

## 🎯 Key Features

- **Attrition Dashboard**: Slice by department, job role, or gender; see KPIs, trends, and top drivers
- **Feature Importance**: Aggregated global importances (e.g., JobRole, OverTime, Age) shown in a color-gradient bar chart
- **Employee Finder**: Sort, filter, and select employees to view risk score and predicted tenure (with survival curve)
- **AI-Powered Insights**: Natural language Q&A for dashboard and employee-level queries (via Gemini LLM)
- **Upskilling Assistant**: Pick an employee, define skill gaps, and get a week-by-week learning plan based on generative AI and real course searches
- **Real Course Links**: All course links in upskilling plans are fetched live from Udemy and Coursera for each skill gap
- **Branding**: Custom banner with right-aligned image/icon (`assets/icon.png`)

---

## 🖼️ Example Banner

![Banner Example](assets/icon.png)  
*Replace this with your app banner or a screenshot if needed.*

---

## 🏗️ Project Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── assets/
│   └── icon.png            # Brand icon/logo (used in banner)
├── .streamlit/
│   └── secrets.toml        # For storing GEMINI_API_KEY (not in repo)
├── README.md
```

---

## 🔒 API Keys

- The Gemini API key must be set via Streamlit secrets:
  - `.streamlit/secrets.toml`  
    ```
    GEMINI_API_KEY = "your-key-here"
    ```
- Do **not** commit your key to GitHub—use Streamlit Cloud “Secrets” settings when deploying.

---

## ⚡ How to Run Locally

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/hr-attrition-upskilling-app.git
   cd hr-attrition-upskilling-app
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your Gemini API key:**  
   Create `.streamlit/secrets.toml` (see above).
4. **Run:**
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ How to Deploy (Streamlit Cloud)

- Push your code to GitHub (don’t add secrets.toml to repo)
- Add your icon image to `assets/icon.png`
- Set your Gemini API Key in Streamlit Cloud “Secrets”
- Deploy and copy the public URL for the app

---

## 💡 Tech Stack

- Streamlit, Python, Pandas, NumPy
- XGBoost (attrition risk)
- Lifelines (Cox PH survival)
- Google Gemini AI (for insights and plan generation)
- Requests, BeautifulSoup (for course search)
- Matplotlib (charts)
- Custom CSS for UI

---

## 🖌️ Customization

- To update the brand icon/banner:  
  Replace `assets/icon.png` and adjust the banner HTML if needed.

---

## 📄 License

[MIT License](LICENSE) *(add if needed)*

---

*Built by [Your Name/Team]. For demo purposes only. No real HR data included.*
