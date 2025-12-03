import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os # Import thư viện os để kiểm tra đường dẫn

# --- KHAI BÁO CÁC GIÁ TRỊ CỐ ĐỊNH ---
LOOK_BACK = 15 
MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'

# --- 1. HÀM TẢI MÔ HÌNH VÀ SCALER (FIX LỖI TẢI FILE) ---
@st.cache_resource # Giúp tải mô hình chỉ một lần
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình hoặc scaler: {e}")
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("⚠ File mô hình hoặc scaler không tồn tại.")
    st.stop()

# --- 2. HÀM DỰ ĐOÁN CHÍNH (Giữ nguyên logic dự đoán) ---
def predict_next_day(input_sequence, model, scaler, look_back):
    
    # 1. Chuẩn hóa dữ liệu đầu vào
    input_reshaped_2d = input_sequence.reshape(-1, 1) 
    input_scaled = scaler.transform(input_reshaped_2d)

    # 2. Định hình lại thành 3D tensor: (1, look_back, 1)
    input_reshaped_3d = input_scaled.reshape(1, look_back, 1) 

    # 3. Dự đoán
    prediction_scaled = model.predict(input_reshaped_3d)

    # 4. Nghịch đảo chuẩn hóa để lấy giá trị gốc
    prediction_original = scaler.inverse_transform(prediction_scaled)

    return prediction_original[0, 0]

# --- 3. GIAO DIỆN STREAMLIT (Giữ nguyên) ---
st.set_page_config(page_title="Hệ thống Dự báo Hành vi Mua sắm (LSTM)", layout="wide")
st.title("🛍️ Hệ thống Dự báo Doanh số Ngày Tiếp theo (LSTM)")

# ... (Phần nhập input và nút bấm giữ nguyên) ...

if st.button("Dự đoán Doanh số Ngày Tiếp theo"):
    input_sequence = np.array(st.session_state[f'input_{i}'] for i in range(LOOK_BACK))
    
    if len(input_sequence) != LOOK_BACK:
        st.error(f"Vui lòng nhập đủ {LOOK_BACK} ngày dữ liệu.")
    else:
        with st.spinner('Đang tính toán dự đoán...'):
            predicted_sales = predict_next_day(input_sequence, model, scaler, LOOK_BACK)
        
        st.success("✅ Dự đoán Hoàn thành!")
        st.balloons()
        
        st.markdown(f"""
            ## Dự đoán Doanh số Ngày Tiếp theo: 
            # <span style='color:green;'>{predicted_sales:,.0f} VNĐ</span>
        """, unsafe_allow_html=True)