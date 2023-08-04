import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df_csv = pd.read_csv("NFL_Python_Filtered.csv")

rush_attempt_map = {rush_attempt: index for index, rush_attempt in enumerate(df_csv["rush_attempt"].unique())}
df_csv["rush_attempt_encoded"] = df_csv["rush_attempt"].map(rush_attempt_map)

feature_set = ['posteam_f', 'quarter_seconds_remaining', 'ydstogo', 'yrdln_f', 'down', 'score_differential_post',
               'defteam_f', 'game_seconds_remaining', 'epa', 'wpa', 'yardline_100', 'half_seconds_remaining',
               'side_of_field_f', 'shotgun', 'drive', 'down_ydstogo', 'posteam_rp_ratio', 'time_score']

X = df_csv[feature_set]
y = df_csv["rush_attempt_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

rf_classifier = RandomForestClassifier(random_state=50)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(cm, index=[f"Actual {rush_attempt}" for rush_attempt in rush_attempt_map],
                            columns=[f"Predicted {rush_attempt}" for rush_attempt in rush_attempt_map])
print(confusion_df)

y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("ROC curve for rush classifier")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.grid(True)
plt.show()

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.2f}")

feature_importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": feature_set, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
