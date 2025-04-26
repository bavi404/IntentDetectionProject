import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Clean text
df['sentence'] = df['sentence'].str.lower().str.strip()

# Encode labels
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])

# Save classes for later decoding
intent_classes = label_encoder.classes_

df.head()
