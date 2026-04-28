# 🚔 Crime Pattern Analysis in India using Machine Learning

This project focuses on analyzing and predicting crime patterns in India using Machine Learning techniques. It helps identify trends, visualize crime distribution, and assist in decision-making for safety and governance. By leveraging historical crime data, the model identifies trends, patterns, and key factors contributing to criminal activities. The goal is to assist in better decision-making, crime prevention strategies, and resource allocation.

---

## 🎯 Objectives
- Analyze crime data across different regions
- Identify patterns and trends
- Build a predictive model for crime classification
- Develop an interactive UI for visualization

---

## 🧠 Features
- 📊 Data preprocessing and cleaning
- 📈 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning model training
- 📉 Model evaluation (accuracy, classification report)
- 💻 Interactive UI using Streamlit

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Jupyter
---


## 📊 Dataset

**File:** `data/crime_dataset_india.csv`

| Column | Description |
|---|---|
| Report Number | Unique ID for each crime report |
| Date Reported | Date the crime was reported |
| Date of Occurrence | Actual date of the crime |
| Time of Occurrence | Time the crime occurred |
| City | City where the crime took place |
| Crime Code | Numeric code for crime type |
| Crime Description | Detailed description of the crime |
| Victim Age | Age of the victim |
| Victim Gender | Gender of the victim |
| Weapon Used | Weapon involved (if any) |
| Crime Domain | Category/domain of the crime (Target Variable) |
| Police Deployed | Number of police personnel deployed |
| Case Closed | Whether the case was closed |
| Date Case Closed | Date the case was closed |

---

## 🧠 ML Pipeline

### 1. Data Preprocessing
- Removed duplicates
- Filled missing values (median for age, "Unknown" for weapons)
- Parsed and extracted date/time features

### 2. Exploratory Data Analysis (EDA)
- Top cities by crime count
- Crime domain distribution
- Yearly crime trends

### 3. Classification (Random Forest)
- **Target:** `Crime Domain`
- **Model:** `RandomForestClassifier` (100 estimators)
- **Split:** 80% train / 20% test
- **Metrics:** Accuracy, Classification Report

### 4. Clustering (K-Means)
- Grouped cities into 5 clusters based on crime count
- Visualized hotspot clusters on a scatter plot

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/aayu04shi/Crime-Anaylis-in-India.git
cd crime-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/ML_Model_CrimeAnalysis.ipynb
```
---

## 📦 Requirements

See `requirements.txt` for the full list. Key libraries:
- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – ML models (Random Forest, K-Means)
- `jupyter` – Notebook environment

---

## 📈 Results

| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| Number of Clusters | 5 (K-Means) |
| Target Variable | Crime Domain |

---

## 📁 Repository Structure

```
Crime-Analysis-in-India/
│
├── app.py
├── leaderboard.csv
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── crime_dataset_india.csv
│
├── models/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── train_model.py
│   ├── data_preprocessing.py
│   └── evaluate.py
│
├── submissions/
│
├── pages/
│   └── leaderboard.py
│
├── .github/
│   └── workflows/
│       └── update_leaderboard.yml
│
├── update_leaderboard_auto.py
├── update_readme.py
```

---
## 📸 Screenshots [User Interface]

### 🔹 Prediction Output
![Prediction](outputs/prediction.png)

### 🔹 Dataset Preview
![Dataset_Preview](outputs/dataset_preview.png)

### 🔹 Crime Analysis Graph
![Graph](outputs/graph.png)


---


## 👥 Contributors

| Name |
|---|
| Sreeya S. S. |
|---|
| Aayushi P. Naik |
|---|
| Saksham S. Lohote |



---

## 📌 How to Participate

Follow these simple steps to contribute to the leaderboard:

### 1️⃣ Fork the Repository

* Click **Fork** (top-right of this repo)
* This creates your own copy

---

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/aayu04shi/Crime-Analysis-in-India.git
cd Crime-Analysis-in-India
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Modify the Model

* Go to:
```bash
src/train_model.py
```
* Change the Model here


---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

###  6️⃣Train the Model

* Enter your **GitHub username** in the app
* Click **"Train / Retrain Model"**

✅ The system will:

* Train the ML model
* Calculate accuracy
* Save your score to the leaderboard

---

### 7️⃣ Commit Your Submission

After training, push your score:

```bash
git add submissions/
git commit -m "Added my model submission"
git pull origin main --rebase
git push
```
---

8️⃣ Create Pull Request
* Go to your fork on GitHub
* Click "Compare & Pull Request"
* Submit PR

---


## 🏆 Leaderboard

👉 Click here to view full leaderboard:
 [View Full Leaderboard](https://crime-analysis-in-india-jtxuwvnkxxfoqocfzhnzex.streamlit.app/)


<!-- LEADERBOARD START -->
Loading leaderboard...
<!-- LEADERBOARD END -->

---



# 🚔 Crime Analysis ML Competition

## 📌 Project Description

Predict the **type of crime** based on time, location, and socio-economic features.  
Submit your predictions as a CSV and get automatically ranked on the leaderboard!

---

## 🗂️ Repository Structure

```
crime-analysis-ml/
├── .github/workflows/grade.yml   ← Auto-grades on every submission push
├── data/
│   ├── train.csv                 ← Training data with labels
│   ├── test.csv                  ← Test data (no labels)
│   └── test.txt                  ← Test set description
├── docs/
│   ├── index.html                ← Live leaderboard page
│   └── leaderboard.json          ← Auto-updated scores
├── grader/
│   └── grader.py                 ← Auto-grading script
├── models/
│   └── baseline_crime_model.pkl  ← Pretrained baseline model
├── starter_code/
│   ├── baseline.py               ← Starter code to get you going
│   └── requirements.txt
├── submission/
│   ├── .gitkeep
│   ├── submission.csv            ← Your submission goes here
│   └── results.csv               ← Auto-generated results
└── README.md
```

---

## 📊 Dataset

### Features (`train.csv` / `test.csv`)

| Column | Description |
|---|---|
| `hour_of_day` | Hour of incident (0–23) |
| `day_of_week` | Day (0=Monday, 6=Sunday) |
| `month` | Month (1–12) |
| `district` | Police district (1–10) |
| `population_density` | People per sq km |
| `unemployment_rate` | Local unemployment % |
| `poverty_rate` | Local poverty % |
| `temperature` | Temperature in °C |
| `is_weekend` | 1 if weekend, else 0 |
| `street_lights` | 1 if lights present, else 0 |
| `prior_incidents` | Prior incidents in area |

### Target

`crime_type` → one of: **Theft, Assault, Burglary, Vandalism, Robbery**

---

## 🚀 How to Participate

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/crime-analysis-ml.git
   cd crime-analysis-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r starter_code/requirements.txt
   ```

3. **Run the baseline**
   ```bash
   cd starter_code
   python baseline.py
   ```

4. **Submit your predictions**
   - Your submission must be a CSV with columns: `id`, `crime_type`
   - Save it as `submission/YourName.csv`
   - Push to GitHub — leaderboard updates automatically!

---

## 🏆 Leaderboard

Live leaderboard → [Leaderboard](https://sssreeya21-ss.github.io/Crime-Analysis/)

Ranked by **Accuracy** on the hidden test set.

---

## 📌 Rules

- Only use `train.csv` for training
- Do **not** hardcode test labels
- One submission file per participant (`YourName.csv`)
- Submissions are graded automatically on push via GitHub Actions

---

## 👨‍💻 Author

Sreeya S S 
Aayushi Naik 
Saksham Lohote
DSBDA Studends MIT-WPU
