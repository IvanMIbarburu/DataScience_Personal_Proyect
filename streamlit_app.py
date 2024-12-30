# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:22:49 2024

@author: USUARIO
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Title
st.title("Real Estate: Search and Price Prediction")
st.header("Property Search")

st.write('This is my first Streamlit app. In the future I want to improve my programming skills.')

data = pd.read_excel("inmuebles_total.xlsx")  # Excel dataframe

data = data.drop('Unnamed: 0', axis = 1)

# Location
location_list = list(data['Location'].unique())
location_list.sort()

area = st.selectbox("Select the Location", ['All'] + list(location_list))

# Type
tipo = st.selectbox("Select type of property", ['All'] + list(data['Type'].unique()))

# Surface
surface_min, surface_max = st.slider("Select surface range (m²)", 
                                           0, 
                                           int(data['Surface (m2)'].max()),
                                           (0, int(data['Surface (m2)'].max())))

# Price
price_min, price_max= st.slider('Select price range (€)',
                                  0,
                                  int(data['Price (€)'].max()),
                                  (0, int(data['Price (€)'].max())),
                                  step = 10000)

# Web:
web = st.selectbox('Select website', ['All'] + list(data['Web'].unique()))



# Filter data
filtered_data = data[(data['Location'] == area if area != 'All' else True) 
                     & (data['Surface (m2)'] >= surface_min)
                     & (data['Surface (m2)'] <= surface_max)
                     & (data['Type'] ==  tipo if tipo != 'All' else True)
                     & (data['Web'] ==  web if web != 'All' else True)
                     & (data['Price (€)'] >= price_min if price_min > 0 else True)
                     & (data['Price (€)'] <= price_max if price_max > 0 else True)]


# Show data
st.write(f'There are: {filtered_data.shape[0]} properties')

st.write('You can download the filtered data if you want!')

st.write("Properties:")
st.dataframe(filtered_data)


# Data histograms:
st.header("Property histogram by location")
hist_data = data
    
if area != 'All':
    hist_data = data[data['Location'] == area]
    
if tipo != 'All':
    hist_data = hist_data[hist_data['Type'] == tipo]

hist_mode = st.selectbox('Histogram visualization', ['overlay', 'stack'])


price_fig = px.histogram(hist_data,
                   title = f'Price histogram of {tipo} in {area}',
                   x = 'Price (€)',
                   color = 'Web',
                   barmode = hist_mode,
                   nbins = 30,
                   color_discrete_sequence=['#00FF00', '#FF0000'])

st.plotly_chart(price_fig)

surface_fig = px.histogram(hist_data,
                   title = f'Surface histogram of {tipo} in {area}',
                   x = 'Surface (m2)',
                   color = 'Web',
                   barmode = hist_mode,
                   nbins = 25,
                   color_discrete_sequence=['#00FF00', '#FF0000'])

st.plotly_chart(surface_fig)



# Predictive model
model = joblib.load("modelo_pisos.pkl")


st.header("Price predictor:")
location = st.selectbox("Select the location", location_list[1:])
surface = st.number_input("Surface (m²)", value=50)



# Button
if st.button("Predict"):
    
    surface_squared = surface ** 2
    
    ohe_location = {f'Location_{loc}':1 if loc == location else 0 for loc in location_list[1:]}
    
    surf_loc = {f'Surface * {loc}': surface * ohe_location[loc] for loc in ohe_location.keys()}
    
    surf_2_loc = {f'Surface squared * {loc}': surface_squared * ohe_location[loc] for loc in ohe_location.keys()}
    
    
    # The model uses 63 coefficients in this order:
    input_piso = {**ohe_location,   # 21 coeff for location (one hot encoded)
                  **surf_loc,       # 21 coeff for surface * location
                  **surf_2_loc      # 21 coeff for surface**2 * location
                  }
    
    
    input_df = pd.DataFrame([input_piso])
    
    output_precio = model.predict(input_df)
    
    st.write(f"Predicted price: {output_precio[0]:.2f} euros")
    
    predictor_fig = px.scatter(data[data['Location'] == location],
                               x = 'Surface (m2)',
                               y = 'Price (€)',
                               color = 'Web',
                               title = f'Prices in {location}',
                               labels = {'Surface (m2)': 'Surface (m2)', 'Price (€)': 'Price (€)'},
                               color_discrete_sequence = ['green', 'red'],)
    
    predictor_fig.add_scatter(x = [surface],
                              y = [output_precio[0]],
                              mode = 'markers',
                              marker = dict(color = 'blue', size = 12),
                              name = 'Predicted')
    
    st.plotly_chart(predictor_fig)

st.write('''**Note**: This app has been created exclusively to learn programming, data analysis, and machine learning.''')
  
