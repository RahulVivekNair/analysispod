import streamlit as st

st.title("Job Queue")
st.write("To view full worker dashboard, click [here](http://localhost:5555/)")
st.components.v1.iframe("http://localhost:5555/tasks", height=800, scrolling=True)