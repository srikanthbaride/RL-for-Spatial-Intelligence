import streamlit as st
import numpy as np

st.title('RL-for-Spatial-Intelligence — Dashboard (Prototype)')

st.markdown('Upload a `q_values.npy` file and inspect basic statistics.')

uploaded = st.file_uploader('q_values.npy', type=['npy'])

if uploaded:
    Q = np.load(uploaded, allow_pickle=False)
    st.write('Q shape:', Q.shape)
    st.write('Q min:', float(np.min(Q)))
    st.write('Q max:', float(np.max(Q)))
    st.line_chart(np.mean(Q, axis=1))
else:
    st.info('No file uploaded yet. Train an agent to produce `artifacts/q_values.npy`.')
