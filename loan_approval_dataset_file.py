import pandas as pd
df = pd.read_csv(r'C:\Users\TAQHI\Desktop\LOAN\loan_approval_dataset1 (1)')
df.isnull().sum()

# Decision Tree Example in Python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
df['self_employed_encoded'] = le.fit_transform(df['self_employed'])


df.drop(columns=['education','self_employed'],axis=1,inplace=True)

X = df.drop(columns=['loan_status'])
y = df['loan_status']


# 2️⃣ Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state= 42)
clf.fit(X, y)

# 3️⃣ Print tree rules (text format)
rules = export_text(clf, feature_names=list(X.columns))
print(rules)

# 4️⃣ Visualize the Decision Tree
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.show()


# 5️⃣ Make Predictions on New Data
new_people = pd.DataFrame({
    'loan_id':[3,15,26,50],
    'Age':[29,49,38,60],
    'no_of_dependents':[3,5,2,1],
    'income':[3000000,5200000,6200000,4200000],
    'loan_amount':[12200000,22200000,32200000,2200000],
    'loan_term':[20,8,13,17],
    'credit score':[367,476,589,789],
    'residential_assets_value':[7200000,3100000,17500000,6100000],
    'commercial_assets_value':[8200000,11500000,3500000,500000],
    'luxury_assets_value':[33300000,8300000,4300000,3300000],
    'bank_asset_value':[12800000,800000	,200000	,20000],
    'education_encoded':[0,1,1,0],
    'self_employed_encoded':[1,0,0,1]
})

predictions = clf.predict(new_people)

print("\n🔮 Predictions for new people:")
for i, pred in enumerate(predictions):
    print(f"Person {i+1}: {pred}")


