# OpenFace Feature Extractor

Production-ready module for extracting facial expression features from exercise videos for exertion prediction using LGBM regression.

## Overview

This module processes exercise videos through OpenFace to extract **167 features** related to facial expressions, which can be used to predict physical exertion levels (RPE - Rate of Perceived Exertion).

## Features Extracted

### 1. **Detection Quality (5 features)**
- Detection rate (% of frames with face detected)
- Detection count
- Confidence statistics (mean, std, min)

### 2. **Action Unit (AU) Statistics (130 features)**
For each of 10 key exertion-related AUs:
- Basic stats: mean, std, max, min, range, median
- Percentiles: 25th, 75th, 90th
- Shape: skewness, kurtosis
- Activation ratio (% time active)
- Peak-to-baseline ratio

**Key AUs for Exertion:**
- AU04: Brow Lowerer (concentration/strain)
- AU06: Cheek Raiser (squinting/strain)
- AU07: Lid Tightener
- AU09: Nose Wrinkler
- AU10: Upper Lip Raiser
- AU12: Lip Corner Puller (grimace)
- AU17: Chin Raiser
- AU20: Lip Stretcher
- AU25: Lips Part (breathing)
- AU26: Jaw Drop (breathing)

### 3. **Repetition Features (8 features)**
- Detected repetition count
- Repetition consistency (how similar each rep is)
- Average repetition intensity
- Repetition timing/tempo
- Detection accuracy (if expected reps provided)

### 4. **Landmark Stability (6 features)**
- Landmark position stability (x, y, overall)
- Head movement statistics (mean, max, std)

### 5. **Temporal Dynamics (12 features)**
- AU velocity (rate of change) for top 3 AUs
- Overall expression change rate

### 6. **Overall AU Activity (4 features)**
- Overall AU mean, max, std across all AUs
- Count of highly active AUs

## Installation

```bash
# Ensure OpenFace is installed and built
cd /path/to/ELEC49X_Group53_AIweightlift/OpenFace
./download_models.sh
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make

# Install Python dependencies
pip install numpy pandas scipy matplotlib opencv-python
```

## Quick Start

### Single Video

```python
from openface_feature_extractor import extract_features

# Extract features
features = extract_features('workout_video.mp4', expected_reps=5)

# Access features
print(f"Detection rate: {features['detection_rate']}")
print(f"Detected reps: {features['detected_reps']}")
print(f"AU04 mean intensity: {features['AU04_mean']}")
```

### Batch Processing

```python
from openface_feature_extractor import extract_features_batch

# Extract from all videos in directory
df = extract_features_batch(
    video_dir='videos/',
    output_csv='training_features.csv',
    expected_reps=5  # or dict: {'video1': 3, 'video2': 5}
)

print(df.head())
```

### Advanced Usage

```python
from openface_feature_extractor import OpenFaceExtractor

# Create extractor instance
extractor = OpenFaceExtractor(verbose=True)

# Show available features
extractor.print_feature_summary()

# Extract features
features = extractor.extract_from_video('video.mp4', expected_reps=5)

# Batch process
df = extractor.extract_from_directory('videos/', output_csv='features.csv')
```

## Integration with LGBM

### Training Pipeline

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from openface_feature_extractor import extract_features_batch

# 1. Extract features from training videos
df = extract_features_batch('training_videos/', output_csv='train_features.csv')

# 2. Load ground truth labels (RPE scores)
# Assuming you have a CSV with video_name and RPE columns
labels = pd.read_csv('rpe_labels.csv')
df = df.merge(labels, on='video_name')

# 3. Prepare features and labels
feature_cols = [col for col in df.columns 
                if col not in ['video_name', 'rpe', 'meta_video_name']]
X = df[feature_cols].values
y = df['rpe'].values

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train LGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# 7. Save model
import pickle
with open('exertion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Prediction Pipeline

```python
from openface_feature_extractor import extract_features
import pickle

# Load trained model
with open('exertion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract features from new video
features = extract_features('new_workout.mp4', expected_reps=5)

# Prepare feature vector (same order as training)
feature_cols = [...]  # Same feature columns used in training
X_new = [features[col] for col in feature_cols]

# Predict exertion level
predicted_rpe = model.predict([X_new])[0]
print(f"Predicted RPE: {predicted_rpe:.1f}")
```

## Video Naming Convention

For automatic rep detection, use these naming patterns:

```
rest_baseline.mp4           # Resting video
squat_low_3reps.mp4         # Low effort, 3 reps
deadlift_medium_5reps.mp4   # Medium effort, 5 reps
bench_high_10reps.mp4       # High effort, 10 reps
```

The extractor will parse:
- **Effort level**: `low`, `medium`, `high`, `rest`
- **Rep count**: `3rep`, `5reps`, etc.

## Feature Selection

With 167 features, you may want to select the most important:

```python
import lightgbm as lgb
import matplotlib.pyplot as plt

# After training
feature_importance = model.feature_importances_
feature_names = feature_cols

# Sort by importance
indices = np.argsort(feature_importance)[::-1]

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(range(20), feature_importance[indices[:20]])
plt.yticks(range(20), [feature_names[i] for i in indices[:20]])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## Output Format

Features are returned as a dictionary or DataFrame with the following structure:

```python
{
    'detection_rate': 0.96,
    'detection_count': 145,
    'confidence_mean': 0.92,
    'AU04_mean': 1.23,
    'AU04_max': 3.45,
    'AU04_std': 0.67,
    'AU04_activation_ratio': 0.45,
    'AU06_mean': 0.89,
    ...
    'detected_reps': 5,
    'rep_consistency': 0.87,
    'rep_avg_intensity': 2.15,
    'landmark_stability_overall': 2.34,
    'head_movement_mean': 3.21,
    'expression_change_mean': 0.45,
    ...
    'metadata': {
        'video_name': 'squat_medium_5reps',
        'width': 1920,
        'height': 1080,
        'fps': 30.0,
        'total_frames': 150,
        'valid_frames': 145
    }
}
```

## Command Line Usage

```bash
# Show feature summary
python3 src/openface_feature_extractor.py

# Extract from single video
python3 src/openface_feature_extractor.py videos/workout.mp4

# Extract from directory
python3 src/openface_feature_extractor.py videos/

# Run examples
python3 examples/feature_extraction_example.py
```

## Troubleshooting

### Low Detection Rate
- Check video quality (resolution, lighting)
- Ensure face is visible throughout video
- Videos should be ≥480p resolution

### Incorrect Rep Detection
- Provide `expected_reps` parameter for validation
- Ensure repetitions have clear exertion patterns
- Adjust `min_distance` in `_detect_peaks()` for faster/slower reps

### Missing Features
- Some features may be 0 if insufficient data
- Requires at least 10 valid frames
- AU features require facial strain/exertion

## Testing

The module has been validated with three test suites:

- **Test A** (`test_a_facedetect.py`): Face detection accuracy ≥95%
- **Test B** (`test_b_landmarks.py`): Landmark stability ≤2px
- **Test C** (`test_c_action_units.py`): AU validation for exertion

Run tests:
```bash
python3 src/test_a_facedetect.py
python3 src/test_b_landmarks.py
python3 src/test_c_action_units.py
```

## Performance

- **Processing time**: ~10-30 seconds per video (depends on length/resolution)
- **Memory usage**: ~200-500 MB per video
- **Recommended**: Batch process overnight for large datasets

## Citation

This module uses OpenFace 2.0:

```
Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency,
OpenFace 2.0: Facial Behavior Analysis Toolkit,
IEEE International Conference on Automatic Face and Gesture Recognition, 2018
```

## License

This project is for academic use (ELEC49X). OpenFace is licensed under the OpenFace Academic License.
