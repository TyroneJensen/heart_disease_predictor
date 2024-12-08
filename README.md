# Heart Disease Predictor

A machine learning web application that predicts the probability of heart disease based on patient data. Built with scikit-learn and Gradio.

## Features

- Interactive web interface for heart disease prediction
- Machine learning model trained on UCI Heart Disease dataset
- Real-time predictions with probability scores
- Simple and intuitive user interface

## Quick Start

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the model training:
```bash
python model.py
```
4. Start the web interface:
```bash
python app.py
```
5. Open http://localhost:7860 in your browser

## Technology Stack

- Python 3.11
- scikit-learn (Machine Learning)
- Gradio (Web Interface)
- Pandas (Data Processing)
- NumPy (Numerical Operations)

## Model Performance

- Algorithm: Logistic Regression
- Accuracy: 87% on test set
- Features: 13 clinical parameters

## Dataset

Uses the UCI Heart Disease dataset with features including:
- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol Levels
- And more...

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
