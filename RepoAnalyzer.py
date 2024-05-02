import streamlit as st
from repo_ai_analyzer import main, main_handler

def init_app():
    logo_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWVIDvMshyWlDWtW0_PdjU_CkGLCBe_9TH8ghjOfWwSA&s'
    st.sidebar.image(logo_url,width=50)
    st.markdown("<h3 style='text-align: center; color: red;'>Hackathon@unlimITed</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>GenAI Repository Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>by</h4>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>FSO Optimization</h3>", unsafe_allow_html=True)
    url = st.sidebar.text_input('Enter Github URL',type='default')
    if url:
        q_c = main_handler(url)
        main(q_c)

if __name__ == "__main__":
    init_app()
    
