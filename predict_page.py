import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)

    return data

data = load_model()

dct = data['model']
le_country = data['le_country']
le_edu = data['le_education']

def show_predict_page():
    st.title('Software Developer Salary Prediction')

    st.write('''''''### We need some information to predict the salary''''''')

    countries = (

        "United States  " ,
        "India"  ,
       " United Kingdom  "  ,
       " Germany    " ,
        "Canada"   ,
       " Brazil    "   ,
      "  France  "  ,
        "Spain  "  ,
        "Australia  "   ,
        "Netherlands" ,
        "Italy"  ,
        "Russian Federation  " ,
        "Sweden ",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor's Degree",
        "Master's Degree"
        "Post Grad"
    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level",education)
    experience = st.slider("Years of Experience",0,50,5)

    ok = st.button("Predict Salary")
    if ok:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.fit_transform(x[:,0])
        x[:, 1] = le_edu.fit_transform(x[:,1])
        x = x.astype(float)

        salary = dct.predict(x)
        st.subheader(f"The estimated Salary is ${salary[0]:.2f}")


