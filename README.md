# 🚨 Anomaly Detection in Financial Transactions

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)]()

A comprehensive machine learning project for detecting fraudulent and anomalous transactions in financial datasets using multiple advanced techniques including traditional ML algorithms and deep learning approaches.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete anomaly detection pipeline for financial transactions, addressing the critical challenge of fraud detection in payment systems. The solution combines exploratory data analysis, preprocessing techniques, and multiple machine learning models to identify suspicious transaction patterns.

**Key Applications:**
- Credit card fraud detection
- Banking transaction monitoring
- Payment system security
- Risk assessment and compliance

## ✨ Features

- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Data Preprocessing**: Handling missing values, scaling, and feature engineering
- **Multiple ML Models**: Isolation Forest, Random Forest, Gradient Boosting, and more
- **Deep Learning**: Autoencoders for unsupervised anomaly detection
- **Model Evaluation**: Detailed performance metrics and comparisons
- **Visualization**: Comprehensive plots and results interpretation

## 📊 Dataset

The project uses financial transaction data with features including:
- Transaction amount
- Merchant information
- Transaction type and category
- Temporal features
- Geographic data (if available)
- Historical patterns

**Data Characteristics:**
- Highly imbalanced classification problem
- Mix of numerical and categorical features
- Real-world noise and missing values

## 🔬 Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Statistical analysis of transaction patterns
- Visualization of feature distributions
- Correlation analysis
- Anomaly pattern identification

### Phase 2: Data Preprocessing
- Handling missing values
- Feature scaling and normalization
- Encoding categorical variables
- Feature engineering and selection

### Phase 3: Machine Learning Modeling
- Training on traditional ML algorithms
- Hyperparameter tuning
- Cross-validation and performance assessment

### Phase 4: Deep Learning
- Autoencoder architecture design
- Reconstruction error analysis
- Anomaly scoring and threshold determination

## 📁 Project Structure

```
Anomaly_Detection_in_Financial_Transactions/
├── 01-EDA-checkpoint.ipynb              # Exploratory Data Analysis
├── 02-Preprocessing-checkpoint.ipynb    # Data Preprocessing
├── 03-Modeling_ML-checkpoint.ipynb      # Machine Learning Models
├── 04-Modeling_ML-checkpoint.ipynb      # Additional ML Analysis
├── 05-Auto-Encoder-checkpoint.ipynb     # Deep Learning Approach
└── README.md                             # Project Documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/saiakhilesh165/Anomaly_Detection_in_Financial_Transactions.git
cd Anomaly_Detection_in_Financial_Transactions
```

2. Install required packages:
```bash
pip install jupyter pandas numpy scikit-learn matplotlib seaborn tensorflow keras
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📖 Usage

Open the notebooks in the following order for best understanding:

1. **01-EDA-checkpoint.ipynb** - Start here to understand the data
2. **02-Preprocessing-checkpoint.ipynb** - Learn about data preparation
3. **03-Modeling_ML-checkpoint.ipynb** - Explore ML algorithms
4. **04-Modeling_ML-checkpoint.ipynb** - Additional model analysis
5. **05-Auto-Encoder-checkpoint.ipynb** - Deep learning approach

Each notebook is self-contained with explanations and visualizations.

## 🤖 Models Implemented

### Traditional Machine Learning
- **Isolation Forest**: Excellent for anomaly detection in high-dimensional spaces
- **Random Forest**: Ensemble method for robust classification
- **Gradient Boosting**: Sequential tree building for improved accuracy
- **Logistic Regression**: Baseline linear classifier
- **Support Vector Machines**: Non-linear classification

### Deep Learning
- **Autoencoders**: Unsupervised learning for anomaly detection using reconstruction error

## 📈 Results

The project provides:
- Model performance comparisons
- ROC-AUC curves and confusion matrices
- Precision-Recall analysis
- Feature importance rankings
- Anomaly detection visualization

## 🔧 Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development

## 📚 Key Libraries

```python
pandas==1.3+
numpy==1.21+
scikit-learn==1.0+
tensorflow==2.8+
matplotlib==3.5+
seaborn==0.11+
```

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Handling imbalanced datasets
- Multiple anomaly detection techniques
- Model evaluation and comparison
- Deep learning implementation
- Data visualization best practices

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sai Akhilesh**
- GitHub: [@saiakhilesh165](https://github.com/saiakhilesh165)

## 📞 Support & Feedback

If you have questions or suggestions:
- Open an issue on GitHub
- Submit a pull request with improvements
- Feel free to reach out with feedback

## 🔗 References

- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [TensorFlow Autoencoders](https://www.tensorflow.org/)
- [Fraud Detection Research Papers](https://arxiv.org/)

---

**Last Updated**: 2026-02-15 03:54:24
**Status**: Active Development