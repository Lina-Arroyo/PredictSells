import streamlit as st
from sklearn.linear_model import LinearRegression
import modelo as modelo

st.title(' 💲Future sells Olist 💲')

st.markdown('This model predict the future sells for year in the ecommerce Olist')

st.header('Calculate the future Sells 💰')

if st.button('Predict the future sells 👀', key='unique'):
    st.markdown('These are the future sells for the next six months on Olist:')
    st.write(modelo.run_model(modelo.train, modelo.test, LinearRegression(), 'LinearRegression'))