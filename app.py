import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.summary import page_summary_body
from app_pages.sale_price_analysis import sale_price_analysis_body
from app_pages.sale_price_predictor import sale_price_predictor_body
from app_pages.project_hypothesis import project_hypothesis_body
from app_pages.predict_price_ml import predict_price_ml_body

# Create an instance of the app
app = MultiPage(app_name="House Sales Predictor")

# Add your app pages here using .add_page()
app.add_page("Project Summary", summary_body)
app.add_page("Sale Price Correlation Analysis", sale_price_analysis_body)
app.add_page("Sale Price Predictor", sale_price_predictor_body)
app.add_page("Project Hypothesis and Validation", project_hypothesis_body)
app.add_page("ML: Price Prediction", predict_price_ml_body)

app.run()  # Run the  app