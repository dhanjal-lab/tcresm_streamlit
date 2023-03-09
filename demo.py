import streamlit as st
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  ### load on cpu if GPU is making issue
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import time
# from PIL import Image

st.set_page_config(page_title="TCR-ESM",page_icon="dna")

hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 2rem;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# image = Image.open('TCR-ESM.png')
# st.image(image)

st.title('TCR-ESM')
st.subheader('a webserver accompanying our work on predicting TCR-peptide-MHC binding with large protein model (ESM1v) embeddings')

dataset = st.radio("Please select the Training Databse",('MCPAS', 'VDJDB'), horizontal=True)

task = st.radio("Please select the Prediction Task",("TCR\u03B1-TCR\u03B2-Peptide-MHC", "TCR\u03B1-TCR\u03B2-Peptide", "TCR\u03B1-Peptide-MHC",
                                                        "TCR\u03B2-Peptide-MHC", "TCR\u03B1-Peptide", "TCR\u03B2-Peptide"), horizontal=True)

with open("sample_input_data.zip", "rb") as file:
    btn = st.download_button(label="Download Sample Input Data",data=file,file_name="sample_input_data.zip", mime="application/octet-stream")
# st.download_button('Download Sample Input Data', open('tcresm_sample_input.zip'))

############## get numpy files
if task == "TCR\u03B1-TCR\u03B2-Peptide-MHC":
    alpha = st.file_uploader("Choose the .npy file containing TCR\u03B1 Embeddings", key=101)
    beta = st.file_uploader("Choose the .npy file containing TCR\u03B2 Embeddings",  key=103)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=109)
    mhc = st.file_uploader("Choose the .npy file containing MHC Embeddings",   key=113)
    shorttask = 'abpm'
    group = (alpha,beta,pepti,mhc)
elif task == "TCR\u03B1-TCR\u03B2-Peptide":
    alpha = st.file_uploader("Choose the .npy file containing TCR\u03B1 Embeddings", key=127)
    beta = st.file_uploader("Choose the .npy file containing TCR\u03B2 Embeddings",  key=131)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=137)
    shorttask = 'abp'
    group = (alpha,beta,pepti)
elif task == "TCR\u03B1-Peptide-MHC":
    alpha = st.file_uploader("Choose the .npy file containing TCR\u03B1 Embeddings", key=139)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=149)
    mhc = st.file_uploader("Choose the .npy file containing MHC Embeddings",   key=151)
    shorttask = 'apm'
    group = (alpha,pepti,mhc)
elif task == "TCR\u03B2-Peptide-MHC":
    beta = st.file_uploader("Choose the .npy file containing TCR\u03B2 Embeddings",  key=157)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=163)
    mhc = st.file_uploader("Choose the .npy file containing MHC Embeddings",   key=167)
    shorttask = 'bpm'
    group = (beta,pepti,mhc)
elif task == "TCR\u03B1-Peptide":
    alpha = st.file_uploader("Choose the .npy file containing TCR\u03B1 Embeddings", key=173)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=179)
    shorttask = 'ap'
    group = (alpha,pepti)
elif task == "TCR\u03B2-Peptide":
    beta = st.file_uploader("Choose the .npy file containing TCR\u03B2 Embeddings",  key=181)
    pepti = st.file_uploader("Choose the .npy file containing Peptide Embeddings", key=191)
    shorttask = 'bp'
    group = (beta,pepti)





##################### ML predict function
@st.cache_data
def predict_on_batch_output(dataset,shorttask,group):

    if dataset == 'MCPAS':
        dataset='mcpas'
    elif dataset== 'VDJDB':
        dataset ='vdjdb'


    if dataset=='mcpas' and shorttask=='abp':
        #load data
        alpha, beta, pep = group
        alpha_np, beta_np, pep_np = np.load(alpha), np.load(beta), np.load(pep)
        #load model
        model = load_model('models/mcpas/bestmodel_alphabetapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, beta_np, pep_np])
    elif dataset=='mcpas' and shorttask=='abpm':
        #load data
        alpha, beta, pep, mhc = group
        alpha_np, beta_np, pep_np, mhc_np = np.load(alpha), np.load(beta), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/mcpas/bestmodel_alphabetaptptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, beta_np, pep_np, mhc_np])
    elif dataset=='mcpas' and shorttask=='ap':
        #load data
        alpha, pep, = group
        alpha_np, pep_np, = np.load(alpha), np.load(pep)
        #load model
        model = load_model('models/mcpas/bestmodel_alphapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np,pep_np])
    elif dataset=='mcpas' and shorttask=='bp':
        #load data
        beta, pep = group
        beta_np, pep_np = np.load(beta), np.load(pep)
        #load model
        model = load_model('models/mcpas/bestmodel_betapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([beta_np, pep_np])
    elif dataset=='mcpas' and shorttask=='apm':
        #load data
        alpha, pep, mhc = group
        alpha_np, pep_np, mhc_np = np.load(alpha), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/mcpas/bestmodel_alphapeptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, pep_np, mhc_np])
    elif dataset=='mcpas' and shorttask=='bpm':
        #load data
        beta, pep, mhc = group
        beta_np, pep_np, mhc_np = np.load(beta), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/mcpas/bestmodel_betapeptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([beta_np, pep_np, mhc_np])
    elif dataset=='vdjdb' and shorttask=='abp':
        #load data
        alpha, beta, pep = group
        alpha_np, beta_np, pep_np = np.load(alpha), np.load(beta), np.load(pep)
        #load model
        model = load_model('models/vdjdb/bestmodel_alphabetapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, beta_np, pep_np])
    elif dataset=='vdjdb' and shorttask=='abpm':
        #load data
        alpha, beta, pep, mhc = group
        alpha_np, beta_np, pep_np, mhc_np = np.load(alpha), np.load(beta), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/vdjdb/bestmodel_alphabetapeptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, beta_np, pep_np, mhc_np])
    elif dataset=='vdjdb' and shorttask=='ap':
        #load data
        alpha, pep, = group
        alpha_np, pep_np, = np.load(alpha), np.load(pep)
        #load model
        model = load_model('models/vdjdb/bestmodel_alphapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, pep_np])
    elif dataset=='vdjdb' and shorttask=='bp':
        #load data
        beta, pep = group
        beta_np, pep_np = np.load(beta), np.load(pep)
        #load model
        model = load_model('models/vdjdb/bestmodel_betapeptide.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([beta_np, pep_np])
    elif dataset=='vdjdb' and shorttask=='apm':
        #load data
        alpha, pep, mhc = group
        alpha_np, pep_np, mhc_np = np.load(alpha), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/vdjdb/bestmodel_alphapeptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([alpha_np, pep_np, mhc_np])
    elif dataset=='vdjdb' and shorttask=='bpm':
        #load data
        beta, pep, mhc = group
        beta_np, pep_np, mhc_np = np.load(beta), np.load(pep), np.load(mhc)
        #load model
        model = load_model('models/vdjdb/bestmodel_betapeptidemhc.hdf5',compile=False)
        #predict_on_batch
        output = model.predict_on_batch([beta_np, pep_np, mhc_np])

    # return np.around(output.squeeze(), 4)

    val = np.squeeze(output)
    return val

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


#####################
if st.button('Submit'):
    # with st.spinner('Wait for it...'):
    #     time.sleep(0.5)
    # res = predict_on_batch_output(dataset,shorttask,group)
    # st.write("Binding Probabilities")
    # st.dataframe((np.round(res, 4)))
    # csv = convert_df(pd.DataFrame(np.round(res, 4), columns=['output']))
    # st.download_button(label="Download Predictions",data=csv,file_name='tcresm_predictions.csv', mime='text/csv')
    try:
        res = predict_on_batch_output(dataset,shorttask,group)
        with st.spinner('Calculating ...'):
            time.sleep(0.5)
            st.write("Binding Probabilities")
            st.dataframe((np.round(res, 4)), use_container_width=500, height=500)
            csv = convert_df(pd.DataFrame(np.round(res, 4), columns=['output']))
            st.download_button(label="Download Predictions",data=csv,file_name='tcresm_predictions.csv', mime='text/csv')
    except:
        st.error('Please ensure you have uploaded the files before pressing the Submit button', icon="🚨")
    


if st.button("Clear All"):
    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.cache_data.clear()



st.caption('Developed By: Shashank Yadav - shashank[at]arizona.edu', unsafe_allow_html=True)