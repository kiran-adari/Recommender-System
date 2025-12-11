# 🎬 Movie Recommendation System

This project implements a complete **movie recommendation system** using classical  
**User–User Collaborative Filtering** with **Cosine Similarity**.  
The system is fully integrated with a **FastAPI backend** and a **React + Vite frontend**,  
delivering real movie recommendations from the **MovieLens 100K dataset**.

---

## 📦 Features Implemented

###  **1. Movie Recommendations**

GET /recommend/{user_id}

Returns the top-N recommended movies for a user based on predicted ratings.

### **2. User Similarity Comparison**

GET /compare?user1=X&user2=Y

Computes cosine similarity between two users’ rating vectors.

---

# 📁 Dataset (MovieLens 100K)

This project uses the **MovieLens 100K** dataset from GroupLens.

### 🔗 Download Dataset:
👉 https://grouplens.org/datasets/movielens/100k/

You only need two files:

| File | Description |
|------|-------------|
| `u.data` | User–item ratings (user_id, item_id, rating) |
| `u.item` | Movie titles |

### 📌 After downloading:
Place the files inside:

backend/
u.data
u.item


⚠️ These files are **NOT included in the repository** due to licensing restrictions.

---

# 🧠 Machine Learning Method

### ✔ 1. Build User–Item Rating Matrix  
A sparse matrix \( R[u,i] \) is created where each row is a user and each column is a movie.

### ✔ 2. Compute User–User Cosine Similarity  

\[
\text{sim}(u,v) = \frac{R_u \cdot R_v}{\|R_u\|\|R_v\|}
\]

### ✔ 3. Predict Missing Ratings  

\[
\hat{r}_{u,i} =
\frac{\sum_{v \in N(u)}\text{sim}(u,v)\, r_{v,i}}
{\sum_{v \in N(u)} |\text{sim}(u,v)|}
\]

### ✔ 4. Recommend Highest Predicted Movies  
The backend returns the top movies with the highest predicted scores.

---

# ⚙️ Backend Setup (FastAPI)

### 📌 Install dependencies
cd backend
pip install -r requirements.txt


### 📌 Start the backend server
uvicorn main:app --reload


Backend runs on:

http://localhost:8000

---

# 💻 Frontend Setup (React + Vite)
cd frontend
npm install
npm run dev


Frontend runs on:

http://localhost:5173

---

# 🧭 Folder Structure
Recommender-System/
|
  |-----backend/
| |------ main.py/
| |------ recommender.py
| |------ attack_experiment.py
| |------ metrics_experiment.py
| |------ u.data (download it from online)
| |------ u.item (download it from online)
|
| |-----frontend/
| |------ src/
| |------ App.jsx
| |------ index.css
| |------ main.jsx
| |------ public/
| |------ package.json
| |------ vite.config.js


---

# 🔌 Available API Endpoints

### 🎯 Get Movie Recommendations
GET /recommend/{user_id}

Example:

GET /recommend/5

### 🔍 Compare Users


---

# 🧱 System Architecture

|-------------------------| HTTP/JSON |--------------------------------|

| React UI | <--------------------------------> | FastAPI |

| ML Engine

User Input Cosine Similarity + CF
|
MovieLens Dataset



---

# 👨‍💻 Author

**Kiran Adari**  
Machine Learning I (16:198:535)  
Professor: Hao Wang  
Rutgers University  

GitHub Repository:  
https://github.com/kiran-adari/Recommender-System

---

# ✔️ Notes

- MovieLens dataset is **not included** in the repo. You must download it manually.








