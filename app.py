import streamlit as st
from sklearn.linear_model import LinearRegression
import modelo as modelo

st.title(' 💲Future sells Olist 💲')

st.markdown('This model predict the future sells for year in the ecommerce Olist')

st.header('Calculate the future Sells 💰')

#path = st.text_input('📥 Enter the CSV path to predict 📥', value="", max_chars=None, key=None, type="default",
                      #help=None, autocomplete=None, on_change=None, 
                      #args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

#sep = st.text_input('📥 Enter the CSV separator  📥', value="", max_chars=None, key=None, type="default",
                      #help=None, autocomplete=None, on_change=None, 
                      #args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")



if st.button('Predict the future sells 👀'):
    st.markdown('These are the future sells for the next six months on Olist:')
    st.write(modelo.run_model(modelo.train, modelo.test, LinearRegression(), 'LinearRegression'))