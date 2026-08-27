
import streamlit as st
import pandas as pd
import numpy as np
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
background-image: url("https://cdn.creazilla.com/digital-illustrations/3224739/light-blue-sky-background-illustration-lg.png");
background-size: cover;
background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
<div style='text-align:center;'>

<h1>📊 Vendor Invoice Intelligence Portal</h1>

<h2>🚀 AI-Driven Freight Cost Prediction & Invoice Risk Flagging</h2>

<p style='font-size:22px;'>
This internal analytics portal leverages machine learning to:
</p>

<p style='font-size:22px;'>
<b>• Forecast freight costs accurately <br>
• Detect risky or abnormal vendor invoices <br>
• Reduce financial leakage and manual workload</b>
</p>

</div>
""", unsafe_allow_html=True)
col1,col2,col3,col4 = st.columns(4)

col1.metric("Avg Freight", "$97.79")
col2.metric("Risk Score", "0.01")
col3.metric("Approval Status", "Normal")
col4.metric("Model Confidence", "94%")


st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---

**Business Impact**
- 📉 Improved cost forecasting  
- 🚨 Reduced invoice fraud & anomalies  
- ⚙️ Faster finance operations  
""")
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About This App"):

    st.markdown("""
### 📊 Vendor Invoice Intelligence System

- Built using Machine Learning + Streamlit

- Models included:
  - Freight Cost Prediction
  - Invoice Risk Flagging

- Supports:
  - Cost Forecasting
  - Manual Approval Detection
  - Risk Analysis Dashboard

---

### 🚚 What is Freight Prediction?

Predicts expected shipping/freight cost
based on invoice quantity and invoice value.

---

### 🚨 What is Invoice Risk Flagging?

Detects whether an invoice should be:

- Auto Approved ✅

- Sent for Manual Review 🚨

based on abnormal invoice patterns.

---

### 🧠 How It Works

- Input invoice data

- ML model processes features

- Predicts freight cost

- Flags risky invoices

- Shows risk score + recommendations

---

### 🛠 Tech Stack

- Python

- Pandas

- Scikit-Learn

- Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🟢 System Status")

st.sidebar.success("Freight Model: Active")
st.sidebar.success("Risk Model: Active")

# --------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**  
    Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**  
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "📦 Quantity",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("📊 Predict Freight Cost")

    if submit_freight:
        # ✅ FIXED (correct feature names)
        input_data = {
            "Quantity": quantity,
            "Dollars": dollars
        }

        prediction = predict_freight_cost(input_data)['Predicted_Freight'].iloc[0]
        st.subheader("🧪 Scenario Simulator")

        if st.button("Simulate 20% Cost Increase"):
            new_dollars = dollars * 1.2
            st.write(f"New invoice dollars: ${new_dollars}")

        st.success("Prediction completed successfully.")

        st.metric(
            label="📦 Estimated Freight Cost",
            value=f"${prediction:,.2f}"

        )
        st.subheader("📜 Prediction History")

        history = pd.DataFrame({
        "Quantity": [quantity],
        "Invoice Dollars": [dollars],
        "Predicted Freight": [prediction]
        })

        st.dataframe(history)
        risk_score = min(prediction / dollars, 1)
        c1, c2, c3 = st.columns(3)

        c1.metric("Predicted Freight", f"${prediction:,.2f}")
        c2.metric("Risk Score", f"{risk_score:.2f}")
        c3.metric("Status", "Normal")
                # 🔥 ADD THIS BELOW METRIC
        st.subheader("📊 Insights")

        st.write(f"• Quantity impact: {quantity}")
        st.write(f"• Invoice value impact: ${dollars}")

        if prediction > 100:
            st.warning("⚠️ High freight cost detected")
        else:
            st.info("✅ Normal freight range")

        

        st.subheader("📊 Risk Meter")

        risk_score = min(prediction / dollars, 1)

        st.progress(risk_score)

        st.write(f"Risk Score: {risk_score:.2f}")
        if risk_score > 0.7:
            st.error("Recommend manual review")

        elif risk_score > 0.4:
            st.warning("Recommend secondary check")

        else:
            st.success("Auto approval recommended")
         
        report = pd.DataFrame({
        "Quantity":[quantity],
        "Dollars":[dollars],
        "Predicted Freight":[prediction]
        })

        st.download_button(
        "Download Report CSV",
        report.to_csv(index=False),
        "prediction_report.csv"
        ) 

        st.subheader("🎯 Model Confidence")

        confidence = 94

        st.progress(confidence/100)

        st.write(f"Model Confidence: {confidence}%")

        st.subheader("🚨 Invoice Risk Gauge")

        if risk_score < 0.30:
            st.success("Low Risk")

        elif risk_score < 0.70:
            st.warning("Medium Risk")

        else:
            st.error("High Risk")
        with st.expander("Model Details"):
            st.write("Freight model: Linear Regression")
            st.write("Invoice risk model: Classifier")

# --------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------
else:
    st.subheader("🚨 Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**  
    Predict whether a vendor invoice should be **flagged for manual approval**  
    based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button("🧾 Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": invoice_quantity,
            "invoice_dollars": invoice_dollars,
            "Freight": freight,
            "total_item_quantity": total_item_quantity,
            "total_item_dollars": total_item_dollars
        }

        flag_prediction = predict_invoice_flag(input_data)["Predicted_Flag"]

        is_flagged = bool(flag_prediction.iloc[0])

        if is_flagged:
            st.error("🚨 Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("✅ Invoice is **SAFE for Auto-Approval**")
col1,col2 = st.columns(2)

with col1:
    st.subheader("🚚 Freight Optimization")
    st.image("https://public.axsmarine.com/wp-content/uploads/2025/08/Port-Call-Optimization_-How-Smart-Data-Cuts-Time-and-Fuel-Waste.png", width=650)
    st.success("Selected for Freight Optimization")

    with st.expander("How it helps"):
        st.write("""
- Predicts freight costs
- Reduces overspending
- Supports budgeting
""")

with col2:
    st.subheader("🚨 Invoice Risk Monitoring")
    st.image("https://img.freepik.com/premium-photo/digital-invoice-document-management-system_1158146-37637.jpg?semt=ais_hybrid&w=740&q=80", width=650)
    st.info("Selected for Invoice Risk Monitoring")
    with st.expander("How it helps"):
        st.write("""
- Flags risky invoices
- Detects anomalies
- Recommends approvals
- Reduces fraud losses                
""")            
       
