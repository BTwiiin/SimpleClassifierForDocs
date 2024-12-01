from cuml.linear_model import LogisticRegression
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.svm import SVC

df = pd.read_csv(r'train_data_with_bboxes.csv')

def extract_bbox_features(bboxes):
    features = []
    for bbox in bboxes:
        if isinstance(bbox, str):
            try:
                coords = json.loads(bbox)
                x_min, y_min = coords[0]
                x_max, y_max = coords[1]
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                aspect_ratio = height / width if width > 0 else 0
                features.append([x_min, y_min, x_max, y_max, width, height, area, aspect_ratio])
            except (json.JSONDecodeError, ValueError):
                features.append([0, 0, 0, 0, 0, 0, 0, 0])
        else:
            features.append([0, 0, 0, 0, 0, 0, 0, 0])
    return np.array(features)

bbox_features = extract_bbox_features(df['Bounding Box'])

scaler = MinMaxScaler()
normalized_bbox_features = scaler.fit_transform(bbox_features)

print(normalized_bbox_features)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
text_embeddings = vectorizer.fit_transform(df['Text']).toarray()

text_embeddings_sparse = csr_matrix(text_embeddings)
bbox_features_sparse = csr_matrix(normalized_bbox_features)

combined_embeddings_sparse = csr_matrix(np.hstack([text_embeddings_sparse.toarray(), bbox_features_sparse.toarray()]))

print(f"Combined embeddings shape: {combined_embeddings_sparse.shape}")

y = df['Label']

X_train, X_val, y_train, y_val = train_test_split(combined_embeddings_sparse, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

print(y_train.dtype)

lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))