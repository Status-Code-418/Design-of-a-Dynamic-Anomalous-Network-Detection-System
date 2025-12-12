import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
import joblib

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# machine Learning Tools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# model Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==============================================================================
# 0. Data Download and Path Setting
# ==============================================================================

print("1. Downloading NSL-KDD dataset...")
# Download latest version
try:
    path = kagglehub.dataset_download("kaggleprollc/nsl-kdd99-dataset")
    print(f"Path to dataset files: {path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("FATAL: Please ensure your Kaggle API credentials are set up correctly.")
    exit() # 종료하여 스크립트 실행 중단


# define column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty_level"
]

# File loading
train_path = os.path.join(path, "KDDTrain+.txt")
test_path = os.path.join(path, "KDDTest+.txt")

train_data = pd.read_csv(train_path, names=column_names, header=None)
test_data = pd.read_csv(test_path, names=column_names, header=None)

print(f"Train Data Shape: {train_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# ==============================================================================
# 1. Exploratory Data Analysis (EDA) & Attack Grouping
# ==============================================================================

print("\n2. Starting EDA and Attack Grouping...")

# Define attack type groups
dos_attacks = ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "mailbomb"]
probe_attacks = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]
r2l_attacks = ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"]
u2r_attacks = ["rootkit", "buffer_overflow", "loadmodule", "perl", "sqlattack", "xterm", "ps"]

# function to categorize attack types
def map_attack_type(attack):
    if attack in dos_attacks:
        return "DoS"
    elif attack in probe_attacks:
        return "Probe"
    elif attack in r2l_attacks:
        return "R2L"
    elif attack in u2r_attacks:
        return "U2R"
    elif attack == "normal":
        return "Normal"
    else:
        return "Other"

# Apply mapping function
train_data["attack_category"] = train_data["attack_type"].apply(map_attack_type)
test_data["attack_category"] = test_data["attack_type"].apply(map_attack_type)

print("\nTrain Attack Category Distribution:")
print(train_data["attack_category"].value_counts())

# ==============================================================================
# 2. Feature Engineering & Feature Selection
# ==============================================================================

print("\n3. Starting Feature Engineering and Selection...")

# Encode Categorical Features
le_map = {} # LabelEncoder 객체를 저장할 딕셔너리
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    le = LabelEncoder()
    # Handle potential unknown categories in test data by fitting only on train data
    train_data[col] = le.fit_transform(train_data[col])
    # The 'handle_unknown' parameter is not explicitly available in simple LabelEncoder.
    # If the test set has unseen labels, it will raise an error if not handled.
    # Given the original notebook uses simple transform, we follow that.
    test_data[col] = le.transform(test_data[col])
    le_map[col] = le # le 객체를 저장

# Encode Attack Categories Numerically
attack_category_encoder = LabelEncoder()
attack_category_encoder.fit(pd.concat([train_data["attack_category"], test_data["attack_category"]], axis=0))
train_data["attack_category_encoded"] = attack_category_encoder.transform(train_data["attack_category"])
test_data["attack_category_encoded"] = attack_category_encoder.transform(test_data["attack_category"])

# Keep a copy of the original dataset for tree-based models
train_data_full = train_data.copy()
test_data_full = test_data.copy()

# Create a separate dataset for feature selection (Linear Models)
train_data_reduced = train_data.copy()
test_data_reduced = test_data.copy()

# Drop non-feature columns
text_cols = ["attack_type", "attack_category", "difficulty_level"]
train_data_reduced.drop(columns=text_cols, inplace=True, errors='ignore')
test_data_reduced.drop(columns=text_cols, inplace=True, errors='ignore')

# Drop the extra column 'attack_status'
train_data_full.drop(columns=['attack_status'], inplace=True, errors='ignore')
test_data_full.drop(columns=['attack_status'], inplace=True, errors='ignore')
train_data_reduced.drop(columns=['attack_status'], inplace=True, errors='ignore')
test_data_reduced.drop(columns=['attack_status'], inplace=True, errors='ignore')


# --- 1: Variance Threshold/Constant Features ---
var_thresh = VarianceThreshold(threshold=0.01)
features_to_check = train_data_reduced.columns.drop('attack_category_encoded', errors='ignore')
var_thresh.fit(train_data_reduced[features_to_check])
low_variance_features = features_to_check[
    ~var_thresh.get_support()
].tolist()

print(f"Features to drop due to low variance/constant: {low_variance_features}")
train_data_reduced.drop(columns=low_variance_features, inplace=True)
test_data_reduced.drop(columns=low_variance_features, inplace=True, errors='ignore')

# --- 2: Correlation Threshold ---
reduced_corr_matrix = train_data_reduced.drop(columns=['attack_category_encoded'], errors='ignore').corr()
correlation_threshold = 0.70
correlated_features = set()
cols = reduced_corr_matrix.columns
for i in range(len(cols)):
    for j in range(i):
        if abs(reduced_corr_matrix.iloc[i, j]) > correlation_threshold:
            correlated_features.add(cols[i])

print(f"Features to drop due to high correlation (> {correlation_threshold}): {correlated_features}")
train_data_reduced.drop(columns=list(correlated_features), inplace=True, errors='ignore')
test_data_reduced.drop(columns=list(correlated_features), inplace=True, errors='ignore')

# --- 3: Variance Inflation Factor (VIF) ---
# Iteratively remove features with high VIF
numeric_cols = [c for c in train_data_reduced.columns if c != 'attack_category_encoded']
vif_threshold = 10
vif_features_to_drop = []

print("VIF check started...")
while True:
    if not numeric_cols or len(numeric_cols) < 2:
        break

    # Recalculate VIF for remaining features
    try:
        vif_values = pd.Series([variance_inflation_factor(train_data_reduced[numeric_cols].values, i)
                                 for i in range(len(numeric_cols))],
                                index=numeric_cols)
    except np.linalg.LinAlgError:
        # Singular matrix error, typically due to perfect correlation introduced by earlier steps
        print("VIF calculation failed (Singular matrix). Stopping VIF check.")
        break
    except ValueError as e:
        # Check if the number of features is too low for VIF calculation
        if "requires at least 2 columns" in str(e):
              break
        else:
              raise e

    max_vif = vif_values.max()

    if max_vif > vif_threshold:
        feature_to_remove = vif_values.idxmax()
        vif_features_to_drop.append(feature_to_remove)
        numeric_cols.remove(feature_to_remove)

        # Drop the feature from the dataset
        train_data_reduced.drop(columns=[feature_to_remove], inplace=True, errors='ignore')
        test_data_reduced.drop(columns=[feature_to_remove], inplace=True, errors='ignore')
    else:
        break

print(f"Features to drop due to VIF > {vif_threshold}: {vif_features_to_drop}")

print("\nFeature Selection Complete (Reduced Dataset)")
print(f"train_data_reduced shape: {train_data_reduced.shape}")
print(f"test_data_reduced shape: {test_data_reduced.shape}")
print(f"train_data_full shape: {train_data_full.shape}")

# ==============================================================================
# 3. Model Training & Evaluation
# ==============================================================================

print("\n4. Starting Model Training and Evaluation...")

# --- 1: Preparing Data for Tree-Based Models ---
X_full_train = train_data_full.drop(columns=["attack_type", "attack_category", "difficulty_level", "attack_category_encoded"], errors="ignore")
y_full_train = train_data_full["attack_category_encoded"]
X_full_test = test_data_full.drop(columns=["attack_type", "attack_category", "difficulty_level", "attack_category_encoded"], errors="ignore")
y_full_test = test_data_full["attack_category_encoded"]

# Ensure columns are aligned
missing_cols = set(X_full_train.columns) - set(X_full_test.columns)
for c in missing_cols:
    X_full_test[c] = 0
X_full_test = X_full_test[X_full_train.columns]

# --- 2: Preparing Data for Linear Models ---
X_reduced_train = train_data_reduced.drop(columns=["attack_category_encoded"], errors="ignore")
y_reduced_train = train_data_reduced["attack_category_encoded"]
X_reduced_test = test_data_reduced.drop(columns=["attack_category_encoded"], errors="ignore")
y_reduced_test = test_data_reduced["attack_category_encoded"]

# Ensure columns are aligned
missing_cols = set(X_reduced_train.columns) - set(X_reduced_test.columns)
for c in missing_cols:
    X_reduced_test[c] = 0
X_reduced_test = X_reduced_test[X_reduced_train.columns]

# --- 3: Apply SMOTE to balance dataset before training all models ---
print("\nApplying SMOTE for data balancing...")
smote = SMOTE(random_state=42)
X_full_train_resampled, y_full_train_resampled = smote.fit_resample(X_full_train, y_full_train)
X_reduced_train_resampled, y_reduced_train_resampled = smote.fit_resample(X_reduced_train, y_reduced_train)

print(f"Full Train Set Resampled Distribution: {Counter(y_full_train_resampled)}")
print(f"Reduced Train Set Resampled Distribution: {Counter(y_reduced_train_resampled)}")

# helper preprocessing function for models that require scaling
def scale_data_set(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# model config keys
RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"
HIST_GRADIENT_BOOSTING_CLASSIFIER = "HistGradientBoostingClassifier"
LOGISTIC_REGRESSION = "LogisticRegression"
ESTIMATOR = "estimator"
DATASET = "dataset"
PRE_PROCESSOR = "preprocessor"
FULL = "full"
REDUCED = "reduced"

# configuration dictionary
model_configs = {
    RANDOM_FOREST_CLASSIFIER: {
        ESTIMATOR: RandomForestClassifier(min_samples_split=10, random_state=42, n_jobs=-1),
        DATASET: FULL
    },
    HIST_GRADIENT_BOOSTING_CLASSIFIER: {
        ESTIMATOR: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42),
        DATASET: FULL
    },
    LOGISTIC_REGRESSION: {
        ESTIMATOR: LogisticRegression(max_iter=2000, solver="newton-cg", random_state=42, n_jobs=-1),
        DATASET: REDUCED,
        PRE_PROCESSOR: scale_data_set
    }
}

# dataset keys
X_TRAIN = "X_train"
Y_TRAIN = "y_train"
X_TEST = "X_test"
Y_TEST = "y_test"

# mapping of dataset types to their corresponding train and test sets
dataset_mapping = {
    FULL: {
        X_TRAIN: X_full_train_resampled,
        Y_TRAIN: y_full_train_resampled,
        X_TEST: X_full_test,
        Y_TEST: y_full_test
    },
    REDUCED: {
        X_TRAIN: X_reduced_train_resampled,
        Y_TRAIN: y_reduced_train_resampled,
        X_TEST: X_reduced_test,
        Y_TEST: y_reduced_test
    }
}

# labels for each class
class_labels = attack_category_encoder.classes_
print(f"Class Labels: {class_labels}")

# train & evaluate
results_summary = []
for model_name, config in model_configs.items():
    print(f"\n==================================================")
    print(f"  Training and Evaluating {model_name}...")
    print(f"==================================================")

    # get the dataset based on the config
    dataset_type = config[DATASET]
    data = dataset_mapping[dataset_type]
    X_train, y_train = data[X_TRAIN], data[Y_TRAIN]
    X_test, y_test = data[X_TEST], data[Y_TEST]

    # apply a model-specific preprocessor if provided
    if PRE_PROCESSOR in config:
        print("Applying StandardScaler...")
        X_train, X_test = config[PRE_PROCESSOR](X_train, X_test)

    # fit and evaluate the model
    model = config[ESTIMATOR]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Using 'target_names' for clear output in classification report
    report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=1, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=1))

    # Add to summary
    results_summary.append({
        "Model": model_name,
        "Dataset": dataset_type,
        "Accuracy": accuracy,
        "Classification_Report": report
    })

    # Print Confusion Matrix for console output
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (Actual Rows vs. Predicted Columns):")
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm_df)

print("\n\n##################################################")
print("               Project Summary")
print("##################################################")

best_accuracy = 0
best_model = ""

for result in results_summary:
    acc = result['Accuracy']
    print(f"\nModel: {result['Model']} (Dataset: {result['Dataset']})")
    print(f"  -> Accuracy: {acc:.4f}")
    print(f"  -> Macro Avg F1: {result['Classification_Report']['macro avg']['f1-score']:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = result['Model']

print(f"\nConclusion: The best performing model was the {best_model} with an accuracy of {best_accuracy:.4f}.")

# ==============================================================================
# 4. Save Model and Preprocessing Components for Real-Time Inference
# ==============================================================================

# 1. Best Model Save (HistGradientBoostingClassifier assumed best)
best_model_name = HIST_GRADIENT_BOOSTING_CLASSIFIER # 또는 'RandomForestClassifier'로 변경 가능
best_model = model_configs[best_model_name][ESTIMATOR]
joblib.dump(best_model, 'nsl_kdd_best_model.pkl')
print(f"\nSaved Best Model ({best_model_name}) to nsl_kdd_best_model.pkl")

# 2. Feature Lists Save
full_features = X_full_train.columns.tolist()
reduced_features = X_reduced_train.columns.tolist()
joblib.dump(full_features, 'nsl_kdd_full_features.pkl')
joblib.dump(reduced_features, 'nsl_kdd_reduced_features.pkl')
print("Saved feature lists.")

# 3. Label Encoder Save
joblib.dump(attack_category_encoder, 'nsl_kdd_label_encoder.pkl')
print("Saved attack category encoder.")

# 4. Scaler Save (Logistic Regression에 사용된 Scaler)
scaler = StandardScaler()
X_reduced_train_original = train_data_reduced.drop(columns=["attack_category_encoded"], errors="ignore")
scaler.fit(X_reduced_train_original)
joblib.dump(scaler, 'nsl_kdd_scaler.pkl')
print("Saved StandardScaler for Reduced Dataset.")

# 5. Categorical Feature Encoders Save
joblib.dump(le_map['protocol_type'], 'nsl_kdd_protocol_le.pkl')
joblib.dump(le_map['service'], 'nsl_kdd_service_le.pkl')
joblib.dump(le_map['flag'], 'nsl_kdd_flag_le.pkl')
print("Saved protocol_type, service, flag encoders.")
