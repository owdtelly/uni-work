import pandas as pd
from helpers import weather_weights
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


### Bring in the three datasets

sd_weather = pd.read_csv("sd_weather_small.csv")

sd_issue_entry = pd.read_csv("sd_issue_entry.csv")

sd_issue_reason = pd.read_csv("sd_issue_reason.csv")


### Merge issue entry on to weather, and then issue reason on to issue entry


data = pd.merge(sd_weather, sd_issue_entry, on="PageEntryId", how="left")

data = pd.merge(data, sd_issue_reason, left_on="IssueReason", right_on="Id", how="left")


### Chnage ObsersationTime to datetime object and extract time from it

data["ObservationTime"] = pd.to_datetime(data.ObservationTime)

data["TimeofObservation"] = (
    data.ObservationTime - data.ObservationTime.dt.normalize()
).dt.total_seconds()

data["ObservationDay"] = data.ObservationTime.dt.day_name()


### Check for duplicates and blanks, then deal with either of them


blanks = data.isna().sum()


duplicates = data.duplicated(
    subset=["PageEntryId"]
)  ## 12543 duplicates in total that need to be dropped

data = data.drop_duplicates(subset=["PageEntryId"])


### Replace blanks in issue column with 0 and replace text with 1


data["IssueOccurred"] = data.Reason.apply(lambda row: 0 if pd.isnull(row) else 1)


### Create the lambda function to apply weather_weights to and change weather description to a float, replace NaN with an average of the weight


data["Weather"] = data.WeatherDescription.apply(
    lambda row: next(
        (key for key, value in weather_weights.items() if row in value), None
    )
)

data = data.fillna(data["Weather"].mean())

### drop unneeded columns, rename Reason to IssueOccurred and reorder columns


columns_to_drop = [
    "PageEntryId",
    "Id",
    "IssueReason",
    "RagStatus",
    "ObservationTime",
    "Reason",
    "WeatherDescription",
    "Reason",
]

clean_data = data.drop(columns=columns_to_drop)

column_names = clean_data.columns.to_list()

column_order = [
    "Weather",
    "ObservationDay",
    "TimeofObservation",
    "CloudCover",
    "Temperature",
    "FeelsLike",
    "Humidity",
    "Precipitation",
    "Pressure",
    "UVIndex",
    "Visibility",
    "WindSpeed",
    "IssueOccurred",
]

clean_data = clean_data[column_order]

### Encode date and time columns

clean_data.ObservationDay = OrdinalEncoder().fit_transform(
    clean_data[["ObservationDay"]]
)


### Check again for nulls

nulls = clean_data.isna().sum()


### Split data into X and y

X = clean_data.iloc[:, :-1].values

y = clean_data.iloc[:, -1].values


### Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


### Scale features apart from weather, date and time

scaler = StandardScaler()

X = scaler.fit_transform(X[:, 3:])


### Train the model on my X data

regressor = LogisticRegression()

regressor.fit(X_train, y_train)


### Predict with the model

prediction = regressor.predict(X_test)


### Check accuracy and classification

accuracy = accuracy_score(y_test, prediction)

classification = classification_report(y_test, prediction)

confusion = confusion_matrix(y_test, prediction)


print(f"Accuracy Score: {accuracy: .2f}")

print("Classification:\n", classification)

print("Confusion Matrix:\n", confusion)


### Adding class weights to account for imbalance between issue being true vs false

class_weights = {0: len(y) / (2 * (len(y) - sum(y))), 1: len(y) / (2 * sum(y))}

weighted_regressor = LogisticRegression(class_weight=class_weights)
weighted_regressor.fit(X_train, y_train)

### Predicting with weighted model

weighted_prediction = weighted_regressor.predict(X_test)


### Check accuracy and classification of weighted model

weighted_accuracy = accuracy_score(y_test, weighted_prediction)

weighted_classification = classification_report(y_test, weighted_prediction)

weighted_confusion = confusion_matrix(
    y_test,
    weighted_prediction,
)


print(f"Weighted Accuracy Score: {weighted_accuracy: .2f}")

print("Weighted Classification:\n", weighted_classification)

print("Weighted Confusion Matrix:\n", weighted_confusion)

### More metrics for checking model

roc_auc = roc_auc_score(y_test, weighted_prediction)

print(roc_auc)

precision, recall, thresholds = precision_recall_curve(y_test, weighted_prediction)

### Produce a plot of the precision and recall

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


### lower threshold of predictor

# custom_prediction = weighted_prediction > 0.3

# custom_accuracy = accuracy_score(y_test, custom_prediction)

# custom_classification = classification_report(y_test, custom_prediction)

# custom_confusion = confusion_matrix(
#     y_test,
#     custom_prediction,
# )

# print(f"Custom Accuracy Score: {custom_accuracy: .2f}")

# print("Custom Classification:\n", custom_classification)

# print("Custom Confusion Matrix:\n", custom_confusion)


ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

params = {
    "scale_pos_weight": ratio,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "seed": 42,
}

# Train XGBoost model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")

print("hello")
