from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_and_prepare_data 

# load data 
x_train, y_train, x_test, y_test = load_and_prepare_data()

# create and train the random forest model
rf_model = RandomForestClassifier(n_estimators= 100, class_weight='balanced', random_state=42, n_jobs=-1)

# train the model on training data
rf_model.fit(x_train, y_train)
print("Model trained successfully.")

# make predictions on the test data
print("Making predictions on the test data...")
y_pred = rf_model.predict(x_test)

# evaluate the model's performance
print("\nEvaluating model performance...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))