import streamlit as st
import joblib
from datetime import datetime, timedelta

# --- App Title ---
st.title("📈 India COVID-19 Forecasting using SARIMA Model")

# --- Sidebar Information ---
st.sidebar.header("🧠 Project Overview")
st.sidebar.markdown("""
- **Project:** COVID-19 Case Forecasting  
- **Model Used:** SARIMA (Seasonal ARIMA)  
- **Training Period:** 2020-01-23 to 2020-08-03  
- **Objective:** Provide meaningful insights to the **Health Department of India**  
to help prepare for **potential future scenarios** based on model predictions.  

---
💻 Project Developed by **Kiran Raj T**   
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('compressed_sarima_model_joblib.pkl')

model = load_model()

# --- Define Training End Date ---
LAST_TRAIN_DATE = datetime(2020, 8, 3).date()

# --- Baseline Cases (for suggestion calculations) ---
# You need to define this based on your dataset; using a placeholder for now
BASELINE_CASES = 100000  # Replace with actual baseline from your dataset

# --- User Input ---
st.subheader("🔮 Forecast Future COVID-19 Cases")
future_date = st.date_input(
    "Select a future date to forecast (after 2020-08-03):",
    min_value=LAST_TRAIN_DATE + timedelta(days=1),
    value=LAST_TRAIN_DATE + timedelta(days=7)
)

# --- Forecast Logic ---
if st.button("Generate Forecast"):
    delta_days = (future_date - LAST_TRAIN_DATE).days

    if delta_days <= 0:
        st.error("⚠️ Please choose a date after the training end date (2020-08-03).")
    else:
        try:
            # Forecast using SARIMA model
            pred = model.forecast(steps=delta_days)
            pred_value = pred[-1]

            st.success(
                f"📢 **Forecast Result:**\n\n"
                f"As per the SARIMA model projection, **India** is expected to record approximately "
                f"**{int(pred_value):,} confirmed COVID-19 cases** on **{future_date.strftime('%d %b %Y')}**."
            )

            # --- Generate Suggestions based on predicted increase ---
            pct_change = ((pred_value - BASELINE_CASES) / BASELINE_CASES) * 100

            st.subheader("🩺 Health Department Suggestions")

            if pct_change <= 10:
                st.info("""
                **Preparedness — Early Warning:**  
                - Continue monitoring cases and maintain testing capacity.  
                - Maintain public hygiene messaging and awareness campaigns.  
                - Check stock levels of PPE, masks, and essential supplies.
                """)
            elif 10 < pct_change <= 50:
                st.warning("""
                **Heightened Response:**  
                - Activate surge capacity plans in hospitals.  
                - Pre-position oxygen concentrators and medicines in predicted hotspots.  
                - Accelerate targeted vaccination and testing campaigns.
                """)
            else:  # pct_change > 50
                st.error("""
                **Urgent / Surge Response:**  
                - Declare localized health emergencies in predicted hotspots.  
                - Open temporary care centers and reallocate medical staff.  
                - Intensify NPIs: mask mandates, social distancing, contact tracing.  
                - Centralize bed management and critical resource allocation.
                """)

            st.info(
                "💡 Note: This forecast is based on historical data trends up to August 3, 2020. "
                "Actual cases may vary depending on real-world factors such as policy changes, public behavior, and virus mutations."
            )
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# --- Footer / Note ---
st.markdown("""
---
📆 **How It Works:**  
1. Pick any future date after **August 3, 2020**.  
2. The app calculates how far ahead to forecast.  
3. It uses the trained **SARIMA model** to predict the corresponding value.  
""")
