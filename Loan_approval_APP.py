# Decision Tree Example in Python   # ORIGINAL
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\TAQHI\Desktop\LOAN\loan_approval_dataset1 (1)')
# Encode categorical features as done before
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
df['self_employed_encoded'] = le.fit_transform(df['self_employed'])


X = df[['loan_id', 'Age', 'no_of_dependents', 'income', 'loan_amount',
       'loan_term', 'credit score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
       'education_encoded', 'self_employed_encoded']]
y = df["loan_status"]

# 2️⃣ Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state= 42)
clf.fit(X, y)

# Streamlit UI
st.title('Loan Approval Predictor (Decision Tree Demo)')
st.write('this is a simple demo using a **Decision Tree** model with dummy data')

# user inputs
age = st.slider('select Age', int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
income = st.number_input('Enter Monthly Income', min_value=int(df['income'].min()), max_value=int(df['income'].max()), value=int(df['income'].min()), step=100)
credit_Score = st.number_input('Enter credit score', min_value=int(df['credit score'].min()), max_value=int(df['credit score'].max()), value=int(df['credit score'].mean()), step=10)
loan_id = st.number_input('Enter Loan ID', min_value=int(df['loan_id'].min()), max_value=int(df['loan_id'].max()), value=int(df['loan_id'].mean()))
no_of_dependents = st.number_input('Enter Number of Dependents', min_value=int(df['no_of_dependents'].min()), max_value=int(df['no_of_dependents'].max()), value=int(df['no_of_dependents'].mean()))
loan_amount = st.number_input('Enter Loan Amount', min_value=int(df['loan_amount'].min()), max_value=int(df['loan_amount'].max()), value=int(df['loan_amount'].mean()))
loan_term = st.number_input('Enter Loan Term', min_value=int(df['loan_term'].min()), max_value=int(df['loan_term'].max()), value=int(df['loan_term'].mean()))
residential_assets_value = st.number_input('Enter Residential Assets Value', min_value=int(df['residential_assets_value'].min()), max_value=int(df['residential_assets_value'].max()), value=int(df['residential_assets_value'].mean()))
commercial_assets_value = st.number_input('Enter Commercial Assets Value', min_value=int(df['commercial_assets_value'].min()), max_value=int(df['commercial_assets_value'].max()), value=int(df['commercial_assets_value'].mean()))
luxury_assets_value = st.number_input('Enter Luxury Assets Value', min_value=int(df['luxury_assets_value'].min()), max_value=int(df['luxury_assets_value'].max()), value=int(df['luxury_assets_value'].mean()))
bank_asset_value = st.number_input('Enter Bank Asset Value', min_value=int(df['bank_asset_value'].min()), max_value=int(df['bank_asset_value'].max()), value=int(df['bank_asset_value'].mean()))
education_encoded = st.selectbox('Education (0: Graduate, 1: Not Graduate)', options=[0, 1])
self_employed_encoded = st.selectbox('Self Employed (0: No, 1: Yes)', options=[0, 1])


# predict button
if st.button('predict loan Approval'):
    new_data = pd.DataFrame([[loan_id, age, no_of_dependents, income, loan_amount, loan_term, credit_Score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, education_encoded, self_employed_encoded]],
                            columns=['loan_id', 'Age', 'no_of_dependents', 'income', 'loan_amount', 'loan_term', 'credit score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'education_encoded', 'self_employed_encoded'])
    prediction = clf.predict(new_data)[0]
    st.success(f'Prediction: Loan {prediction}')


# show the decision tree diagram
st.subheader('Decision Tree Visualization')

fig,ax = plt.subplots(figsize=(10,6))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
st.pyplot(fig)
# show tree rules
st.subheader('Decision Tree Rules (Text Form)')
rules = export_text(clf,feature_names=list(X.columns))
st.text(rules)