# Heart Disease Predictor - Detailed Documentation

## Project Overview

This project implements a machine learning model to predict heart disease probability using patient clinical data. It features a web interface for easy interaction and real-time predictions.

## Technical Implementation

### 1. Data Processing (`model.py`)

The project uses the UCI Heart Disease dataset with the following features:

| Feature | Description | Values |
|---------|-------------|---------|
| age | Age in years | 29-77 |
| sex | Gender | 0 = female, 1 = male |
| cp | Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| trestbps | Resting blood pressure (mm Hg) | 94-200 |
| chol | Serum cholesterol (mg/dl) | 126-564 |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| restecg | Resting ECG results | 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy |
| thalach | Maximum heart rate achieved | 71-202 |
| exang | Exercise induced angina | 0 = no, 1 = yes |
| oldpeak | ST depression induced by exercise relative to rest | 0-6.2 |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels colored by fluoroscopy | 0-3 |
| thal | Thalassemia | 3 = normal, 6 = fixed defect, 7 = reversible defect |

### 2. Model Training

- Algorithm: Logistic Regression
- Train-Test Split: 80-20
- Feature Scaling: StandardScaler
- Model Performance: 87% accuracy on test set

### 3. Web Interface (`app.py`)

Built using Gradio with:
- Input fields for all 13 features
- Real-time predictions
- Probability scores for both classes

## Test Cases

### 1. High Risk Patient
```
Age: 65
Sex: 1 (male)
Chest Pain Type: 4 (asymptomatic)
Resting Blood Pressure: 160
Cholesterol: 286
Fasting Blood Sugar > 120: 1 (true)
Resting ECG: 2
Maximum Heart Rate: 108
Exercise Induced Angina: 1 (yes)
ST Depression: 2.5
Slope of ST Segment: 2
Number of Major Vessels: 3
Thal: 7
Expected: High probability of heart disease
```

### 2. Low Risk Patient
```
Age: 40
Sex: 0 (female)
Chest Pain Type: 1
Resting Blood Pressure: 120
Cholesterol: 200
Fasting Blood Sugar > 120: 0
Resting ECG: 0
Maximum Heart Rate: 160
Exercise Induced Angina: 0
ST Depression: 0.5
Slope of ST Segment: 1
Number of Major Vessels: 0
Thal: 3
Expected: Low probability of heart disease
```

### 3. Moderate Risk Patient
```
Age: 52
Sex: 1 (male)
Chest Pain Type: 2
Resting Blood Pressure: 140
Cholesterol: 245
Fasting Blood Sugar > 120: 0
Resting ECG: 1
Maximum Heart Rate: 142
Exercise Induced Angina: 1
ST Depression: 1.2
Slope of ST Segment: 1
Number of Major Vessels: 1
Thal: 6
Expected: Moderate probability of heart disease
```

## Project Structure

```
heart_disease_predictor/
├── model.py           # Data processing and model training
├── app.py            # Gradio web interface
├── requirements.txt  # Project dependencies
├── README.md        # Project overview
└── DOCUMENTATION.md # Detailed documentation
```

## Development Process

1. Data Collection and Analysis
   - Loaded UCI Heart Disease dataset
   - Performed basic EDA
   - Handled missing values

2. Model Development
   - Preprocessed features
   - Split data into training and test sets
   - Trained logistic regression model
   - Evaluated model performance

3. Web Interface Development
   - Created Gradio interface
   - Added input validation
   - Implemented real-time predictions

## Future Improvements

1. Model Enhancements
   - Try different algorithms (Random Forest, XGBoost)
   - Feature engineering
   - Hyperparameter tuning

2. Interface Improvements
   - Add data visualization
   - Include feature importance analysis
   - Add batch prediction capability

3. Additional Features
   - Save prediction history
   - Export predictions to CSV
   - Add user authentication

## Troubleshooting

Common issues and solutions:

1. Missing Dependencies
```bash
pip install -r requirements.txt
```

2. Model File Not Found
```bash
# Run model training first
python model.py
```

3. Port Already in Use
```bash
# Change port in app.py
iface.launch(server_port=7861)
```
