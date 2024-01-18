import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title='Intrusion Detection Dashboard', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')

hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            background:dodgerblue;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """
    
sub_title = """
            <div>
                <h6 style="color:dodgerblue;
                text-align:center;
                margin-top:-40px;">
                Intrusion Detection Dashboard </h6>
            </div>
            """

st.markdown(sub_title,
            unsafe_allow_html=True)

screen = st.empty()

# Load the trained model
from joblib import load

# Load the trained model
model = load("trained_model.joblib")

# Define the input features
input_features = ["protocol_type", "flag", "src_bytes", "dst_bytes", "count",
                  "same_srv_rate", "diff_srv_rate", "dst_host_srv_count",
                  "dst_host_same_srv_rate", "dst_host_same_src_port_rate"]

# Define the label encoder
label_encoder = LabelEncoder()



# Add a sidebar to the app
st.sidebar.header("User Input Features")

# Add input fields to the sidebar
protocol_type = st.sidebar.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
flag = st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO"])
src_bytes = st.sidebar.number_input("Source Bytes", 0, 5000, 2500)
dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 5000, 2500)
count = st.sidebar.number_input("Count", 0, 100, 50)
same_srv_rate = st.sidebar.number_input("Same Service Rate", 0.0, 1.0, 0.5)
diff_srv_rate = st.sidebar.number_input("Different Service Rate", 0.0, 1.0, 0.5)
dst_host_srv_count = st.sidebar.number_input("Destination Host Service Count", 0, 100, 50)
dst_host_same_srv_rate = st.sidebar.number_input("Destination Host Same Service Rate", 0.0, 1.0, 0.5)
dst_host_same_src_port_rate = st.sidebar.number_input("Destination Host Same Source Port Rate", 0.0, 1.0, 0.5)

# Label encode the input features
label_encoded_input_features = label_encoder.fit_transform(input_features)





# Create a numpy array with the input features
input_array = np.array([
    protocol_type, flag, src_bytes, dst_bytes, count,
    same_srv_rate, diff_srv_rate, dst_host_srv_count,
    dst_host_same_srv_rate, dst_host_same_src_port_rate
])

# Fit the label encoder to the input data
label_encoder.fit(input_array)

# Transform the input data to numerical values
input_array = label_encoder.transform(input_array)


# Define a function to make a prediction
def predict():
    # Create a Streamlit app
     
    st.title("Network Intrusion Detection System")
    st.markdown("""The Intrusion Detection System (IDS) developed using a Gradient Boosting Classifier to classify network activities into normal and intrusive instances. The model was trained and fine-tuned using a diverse dataset while using **Streamlit** as a GUI for user inputs""")


    
    # Make a prediction using the trained model
    prediction = model.predict(input_array.reshape(1,-1))
    pred = model.predict_proba(input_array.reshape(1, -1))
    # Add a section to display the input features details
    st.header(" Input Features:")

# Print out the input features details
    st.write(f"Protocol Type: {protocol_type}")
    st.write(f"Flag: {flag}")
    st.write(f"SRC Bytes: {src_bytes}")
    st.write(f"DST Bytes: {dst_bytes}")
    st.write(f"Count: {count}")
    st.write(f"Same Service Rate: {same_srv_rate}")
    st.write(f"Different Service Rate: {diff_srv_rate}")
    st.write(f"Destination Host Service Count: {dst_host_srv_count}")
    st.write(f"Destination Host Same Service Rate: {dst_host_same_srv_rate}")
    st.write(f"Destination Host Same Source Port Rate: {dst_host_same_src_port_rate}")


    # Print the prediction
    
    predicted_class = "Normal" if prediction[0] == 1 else "Anomaly"
    
    
    st.header(f"The predicted class is: {predicted_class}")
    
    


# Call the predict function when the "Predict" button is clicked
st.sidebar.button("Predict", on_click=predict)
