import pandas as pd
import numpy as np
import streamlit as st
import joblib

@st.cache
def load_data():
    return pd.read_csv('data/advertising_regression.csv')

# Title of your Web Application
st.title('Sales Forecasting')

# Describe your Web Application
st.write('We demonstrate how we can forecast advertising sales based on ad expenditure.')

st.subheader('Explore Data')
# Read Data
data = load_data()
cols = ["TV", "radio", "newspaper", "sales"]
st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)
# Show Data
data[st_ms]

# Create Sidebar
# Sidebar Description
st.sidebar.subheader('Advertising Costs')

# TV Sidebar
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)
# Radio Sidebar
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)
# Newspaper Sidebar
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)
# Sales
predicted_sale_slot = st.sidebar.empty()

st.subheader('View Distribution Graphs')
option = st.selectbox('Select graph', options=['Radio', 'TV', 'Newspaper'])
display_all = st.checkbox('View All')

if option == 'Radio' or display_all:
    # Radio
    st.subheader('Radio Ad Cost Distribution')
    # Distribution of Radio Advertising Cost
    hist_values_radio = np.histogram(data.radio, bins=300, range=(0, 300))[0]\
    # Show Bar Chart
    st.bar_chart(hist_values_radio)
if option == 'TV' or display_all:
    # TV
    st.subheader('TV Ad Cost Distribution')
    # Distribution of TV Advertising Cost
    hist_values_tv = np.histogram(data.TV, bins=300, range=(0, 300))[0]
    # Show Bar Chart
    st.bar_chart(hist_values_tv)
if option == 'Newspaper' or display_all:
    # Newspaper
    st.subheader('Newspaper Ad Cost Distribution')
    # Distribution of Newspaper Advertising Cost
    hist_values_newspaper = np.histogram(data.newspaper, bins=300, range=(0, 300))[0]
    # Show Bar Chart
    st.bar_chart(hist_values_newspaper)

# Load saved maching learning model
saved_model = joblib.load('advertising_model.sav')

# Predict sales using variables/features
predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0]
predicted_sale_slot.markdown(f'Predicted sales is\n**${predicted_sales}**.')


