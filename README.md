# Insurance AI Intelligence System

An AI-powered system designed to analyze insurance leads, predict conversion propensity, and identify at-risk customers for churn prevention. Built with FastAPI, SQLite, Pandas, and Scikit-Learn (XGBoost), with a vanilla HTML/CSS/JS frontend dashboard.

## Features

- **Lead Propensity Scoring**: Predicts the likelihood of a lead converting into a customer using an XGBoost model.
- **Customer Churn Prediction**: Identifies customers at high risk of canceling their policies.
- **NLP Sentiment Analysis**: Analyzes customer feedback to detect positive or negative sentiment.
- **Explainable AI (SHAP)**: Provides top contributing factors for why a lead scored high or a customer is at risk.
- **Robust Data Ingestion**: Automatically cleans and normalizes messy datasets (like Kaggle CSVs) with intelligent column mapping and auto-generation for missing IDs.
- **Interactive Dashboard**: A dynamic glassmorphism UI for viewing analytics, managing CSV uploads, and triggering on-demand model retraining.

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy (SQLite), Pandas
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP, TextBlob
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Server**: Uvicorn

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd insurance_ai_system
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Access the Dashboard**
   Open your browser and navigate to: [http://localhost:8000/ui/](http://localhost:8000/ui/)

## Usage

1. **Upload Data**: Navigate to the "Data Management" tab to upload CSV datasets for Leads or Customers. The system handles messy Kaggle data automatically.
2. **Train Models**: Click "Retrain Models" to trigger a fresh XGBoost training pipeline on the latest database records.
3. **View Insights**: The "Overview" and specific directory tabs provide real-time AI scoring, primary risk factors, and NLP sentiment analysis.

## Project Structure

```
├── app/
│   ├── api/            # FastAPI Routers
│   ├── config/         # App configuration
│   ├── database/       # SQLAlchemy models and schemas
│   ├── models/         # Machine Learning algorithms
│   ├── training/       # AI Training orchestration
│   └── uploads/        # Data ingestion & cleaning engine
├── frontend/           # HTML, CSS, JS Dashboard
├── requirements.txt
└── docker-compose.yml  
```
