# Visa Status Predictor & Processing Time Estimator 🛂

A full‑stack machine learning web application built with **Flask**, **SQLAlchemy**, and **PostgreSQL**, deployed on **Render**.  
The app predicts visa approval probability and processing time based on applicant details, while providing context‑aware suggestions.  
It also includes secure user authentication with role‑based access (user/admin) and persistent storage.

---

## 🚀 Live Demo
[Visa Status Predictor](https://visa-status-predictor.onrender.com)

---

## ✨ Features
- **User Authentication**  
  - Signup/Login with hashed passwords  
  - Role‑based access (user vs admin)  
  - Session management with Flask  

- **Visa Prediction**  
  - Predicts approval probability using a trained classifier  
  - Estimates realistic processing time using regression models  
  - Provides personalized suggestions based on applicant profile  

- **Admin Dashboard**  
  - View all predictions with detailed logs  
  - Statistics by education, continent, and visa status  
  - Trend charts and average processing time  

- **Persistent Database**  
  - PostgreSQL on Render for user accounts and predictions  
  - SQLAlchemy ORM for clean and maintainable queries  

---

## 🛠️ Tech Stack
- **Backend:** Flask, SQLAlchemy  
- **Database:** PostgreSQL (Render)  
- **ML Models:** scikit‑learn (Random Forest, Linear Regression)  
- **Frontend:** HTML, CSS  
- **Deployment:** Render (Python 3 runtime)  

---


