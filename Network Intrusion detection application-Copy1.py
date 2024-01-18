#!/usr/bin/env python
# coding: utf-8

# # Imports & Configurations

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn import tree
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
import itertools
from tabulate import tabulate
import os
warnings.filterwarnings('ignore')


# # Data Preprocessing & EDA

# In[ ]:


train=pd.read_csv('T//Train_data.csv')
test=pd.read_csv('T//Test_data.csv')
train


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.describe(include='object')


# ## Missing Data

# In[ ]:


total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")


# No missing values

# ## Duplicates 

# In[ ]:


print(f"Number of duplicate rows: {train.duplicated().sum()}")


# Great! No duplicates

# ## Outliers 

# In[ ]:


plt.figure(figsize=(40,30))
sns.heatmap(train.corr(), annot=True)

# import plotly.express as px
# fig = px.imshow(df.corr(), text_auto=True, aspect="auto")
# fig.show()


# In[ ]:


sns.countplot(x=train['class'])


# # Label Encoding

# In[ ]:


def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)


# In[ ]:


train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)
train.head()


# # Feature selection

# In[ ]:


X_train = train.drop(['class'], axis=1)
Y_train = train['class']


# In[ ]:


rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features


# In[ ]:


X_train = X_train[selected_features]


# # Split and scale data

# In[ ]:


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)


# In[ ]:


def objective(trial):
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=False)
    rf_max_features = trial.suggest_int('rf_max_features', 2, 10, log=False)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 3, 20, log=False)
    classifier_obj = RandomForestClassifier(max_features = rf_max_features, max_depth = rf_max_depth, n_estimators = rf_n_estimators)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy


# In[ ]:


study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective, n_trials=30)
print(study_rf.best_trial)


# In[ ]:


rf = RandomForestClassifier(max_features = study_rf.best_trial.params['rf_max_features'], max_depth = study_rf.best_trial.params['rf_max_depth'], n_estimators = study_rf.best_trial.params['rf_n_estimators'])
rf.fit(x_train, y_train)

rf_train, rf_test = rf.score(x_train, y_train), rf.score(x_test, y_test)

print(f"Train Score: {rf_train}")
print(f"Test Score: {rf_test}")


# In[ ]:


from matplotlib import pyplot as plt

def f_importance(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('feature importance for RF')
    plt.show()

# whatever your features are called
features_names = selected_features

# Specify your top n features you want to visualize.
# You can also discard the abs() function 
# if you are interested in negative contribution of features
f_importance(abs(rf.feature_importances_), features_names, top=7)


# # SKLearn Gradient Boosting Model

# In[ ]:


SKGB = GradientBoostingClassifier(random_state=42)
SKGB.fit(x_train, y_train)


# In[ ]:


SKGB_train, SKGB_test = SKGB.score(x_train , y_train), SKGB.score(x_test , y_test)

print(f"Training Score: {SKGB_train}")
print(f"Test Score: {SKGB_test}")


# In[ ]:


from sklearn.metrics import confusion_matrix

y_pred = SKGB.predict(x_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Output the confusion matrix
print(cm)


# In[ ]:


import joblib

# Save the model to a file
joblib.dump(SKGB, "trained_model.joblib")


# # SKLearn AdaBoost Model

# In[ ]:


ab_model = AdaBoostClassifier(random_state=42)


# In[ ]:


ab_model.fit(x_train, y_train)


# In[ ]:


ab_train, ab_test = ab_model.score(x_train , y_train), ab_model.score(x_test , y_test)

print(f"Training Score: {ab_train}")
print(f"Test Score: {ab_test}")


# # Summary

# In[ ]:


data = [["AdaBoost", ab_train, ab_test],
        ["GBM", SKGB_train, SKGB_test],
]

col_names = ["Model", "Train Score", "Test Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))


# In[ ]:


get_ipython().run_cell_magic('writefile', 'network_model.py', 'import streamlit as st\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.ensemble import RandomForestClassifier\n\n\nst.set_page_config(page_title=\'Intrusion Detection Dashboard\', \n                       layout = \'wide\', \n                       initial_sidebar_state = \'auto\')\n\nhide_menu_style = """\n    <style>\n        MainMenu {visibility: hidden;}\n        \n        \n         div[data-testid="stHorizontalBlock"]> div:nth-child(1)\n        {  \n            border : 2px solid #doe0db;\n            border-radius:5px;\n            text-align:center;\n            color:black;\n            background:dodgerblue;\n            font-weight:bold;\n            padding: 25px;\n            \n        }\n        \n        div[data-testid="stHorizontalBlock"]> div:nth-child(2)\n        {   \n            border : 2px solid #doe0db;\n            background:dodgerblue;\n            border-radius:5px;\n            text-align:center;\n            font-weight:bold;\n            color:black;\n            padding: 25px;\n            \n        }\n    </style>\n    """\n    \nsub_title = """\n            <div>\n                <h6 style="color:dodgerblue;\n                text-align:center;\n                margin-top:-40px;">\n                Intrusion Detection Dashboard </h6>\n            </div>\n            """\n\nst.markdown(sub_title,\n            unsafe_allow_html=True)\n\nscreen = st.empty()\n\n# Load the trained model\nfrom joblib import load\n\n# Load the trained model\nmodel = load("trained_model.joblib")\n\n# Define the input features\ninput_features = ["protocol_type", "flag", "src_bytes", "dst_bytes", "count",\n                  "same_srv_rate", "diff_srv_rate", "dst_host_srv_count",\n                  "dst_host_same_srv_rate", "dst_host_same_src_port_rate"]\n\n# Define the label encoder\nlabel_encoder = LabelEncoder()\n\n\n\n# Add a sidebar to the app\nst.sidebar.header("User Input Features")\n\n# Add input fields to the sidebar\nprotocol_type = st.sidebar.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])\nflag = st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO"])\nsrc_bytes = st.sidebar.number_input("Source Bytes", 0, 5000, 2500)\ndst_bytes = st.sidebar.number_input("Destination Bytes", 0, 5000, 2500)\ncount = st.sidebar.number_input("Count", 0, 100, 50)\nsame_srv_rate = st.sidebar.number_input("Same Service Rate", 0.0, 1.0, 0.5)\ndiff_srv_rate = st.sidebar.number_input("Different Service Rate", 0.0, 1.0, 0.5)\ndst_host_srv_count = st.sidebar.number_input("Destination Host Service Count", 0, 100, 50)\ndst_host_same_srv_rate = st.sidebar.number_input("Destination Host Same Service Rate", 0.0, 1.0, 0.5)\ndst_host_same_src_port_rate = st.sidebar.number_input("Destination Host Same Source Port Rate", 0.0, 1.0, 0.5)\n\n# Label encode the input features\nlabel_encoded_input_features = label_encoder.fit_transform(input_features)\n\n\n\n\n\n# Create a numpy array with the input features\ninput_array = np.array([\n    protocol_type, flag, src_bytes, dst_bytes, count,\n    same_srv_rate, diff_srv_rate, dst_host_srv_count,\n    dst_host_same_srv_rate, dst_host_same_src_port_rate\n])\n\n# Fit the label encoder to the input data\nlabel_encoder.fit(input_array)\n\n# Transform the input data to numerical values\ninput_array = label_encoder.transform(input_array)\n\n\n# Define a function to make a prediction\ndef predict():\n    # Create a Streamlit app\n     \n    st.title("Network Intrusion Detection System")\n    st.markdown("""The Intrusion Detection System (IDS) developed using a Gradient Boosting Classifier to classify network activities into normal and intrusive instances. The model was trained and fine-tuned using a diverse dataset while using **Streamlit** as a GUI for user inputs""")\n\n\n    \n    # Make a prediction using the trained model\n    prediction = model.predict(input_array.reshape(1,-1))\n    pred = model.predict_proba(input_array.reshape(1, -1))\n    # Add a section to display the input features details\n    st.header(" Input Features:")\n\n# Print out the input features details\n    st.write(f"Protocol Type: {protocol_type}")\n    st.write(f"Flag: {flag}")\n    st.write(f"SRC Bytes: {src_bytes}")\n    st.write(f"DST Bytes: {dst_bytes}")\n    st.write(f"Count: {count}")\n    st.write(f"Same Service Rate: {same_srv_rate}")\n    st.write(f"Different Service Rate: {diff_srv_rate}")\n    st.write(f"Destination Host Service Count: {dst_host_srv_count}")\n    st.write(f"Destination Host Same Service Rate: {dst_host_same_srv_rate}")\n    st.write(f"Destination Host Same Source Port Rate: {dst_host_same_src_port_rate}")\n\n\n    # Print the prediction\n    \n    predicted_class = "Normal" if prediction[0] == 1 else "Anomaly"\n    \n    \n    st.header(f"The predicted class is: {predicted_class}")\n    \n    \n\n\n# Call the predict function when the "Predict" button is clicked\nst.sidebar.button("Predict", on_click=predict)\n')


# In[ ]:


get_ipython().system('streamlit run network_model.py')


# In[ ]:




