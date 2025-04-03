import streamlit as st

st.title("Hello, Streamlit! 👋")

name = st.text_input("What's your name?", "Abdullah")
age = st.slider("Select your age", 0, 100, 25)

st.write(f"Hello, **{name}**! You are **{age}** years old.")

if st.button("Say Hi"):
    st.success(f"Hi {name}! 🎉")
