# 🔍 Fake News Detection Using Content and Source Analysis

![GitHub stars](https://img.shields.io/github/stars/yourusername/fake-news-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/fake-news-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/fake-news-detection?style=social)

> A machine learning approach to detect fake news by analyzing both content and source credibility, achieving state-of-the-art performance.

## 📋 Overview

This project presents an enhanced approach to fake news detection by combining content analysis with source credibility evaluation. Using an `AdaBoost classifier` trained on features extracted from both article text and URLs, the model achieves high performance on a source-based fake news dataset from [Kaggle](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification).

## 🌟 Key Contributions

### 1. Enhanced Feature Engineering 🧪

This project goes beyond basic content analysis by extracting rich features from both the news articles and their sources:

#### Text Processing Pipeline:

- **Cleaning & Normalization**: Lowercase conversion, HTML tag removal, and elimination of unwanted characters
- **Advanced Tokenization**: Language-specific processing using `spaCy` models for both English and German text
- **Lemmatization**: Maintaining linguistic context across different languages
- **TF-IDF Vectorization**: Separate vectorizers for content and source data

#### URL-Based Features:

```python
def clean_site_url(url):
    url = url.lower()
    url = re.sub(r'\d+', '', url)
    tokens = re.split(r'[./?=_-]+', url)
    tokens = [token for token in tokens if token]
    return ' '.join(tokens)

def clean_img_url(url):
    url = url.lower()
    url = re.sub(r'https?://(www\.)?', '', url)
    url = re.sub(r'\d+', '', url)    
    url = re.sub(r'\.?(\bjpg\b|\bjpeg\b|\bpng\b|\bgif\b|\bbmp\b)', '', url)    
    unwanted_words = ['uploads', 'content', 'images', 'files', 'fullscreen', 'capture']
    for word in unwanted_words:
        url = url.replace(word, '')    
    tokens = re.split(r'[./?=%_\-]+', url)
    tokens = [token for token in tokens if token]
    return ' '.join(tokens[:10]) # Only return the first 10 tokens to reduce noise 
```

The URL processing extracts informative tokens that represent source credibility aspects:
- Article/image titles embedded in URLs
- Domain extensions (which can indicate credibility)
- URL structure and patterns

### 2. Class Imbalance Handling ⚖️

The dataset showed a significant imbalance between real and fake news articles:

<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/bdd19097-e91a-49ad-a0f0-1448094e19f6" alt="Class Distribution" width="300"/></td>
    <td><img src="https://github.com/user-attachments/assets/13d9c96e-b247-4c01-8406-a5902d3a5896" alt="Distribution Chart" width="300"/></td>
  </tr>
</table>

To address this imbalance, **Synthetic Minority Over-sampling Technique (SMOTE)** was applied:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)
```

Before and after SMOTE application:
```
Before SMOTE:
0 (Real)    890
1 (Fake)    542

After SMOTE:
0 (Real)    890
1 (Fake)    890
```

### 3. Model Architecture 🏗️

The project employs an **Ensemble Learning** approach with **AdaBoost** using **Decision Trees** as base estimators:

```python
# Initialize the AdaBoost classifier with decision tree as the base estimator
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=10),
    n_estimators=5,
    algorithm='SAMME',
    learning_rate=1.0,
    random_state=42
)

# Train the classifier
ada_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = ada_clf.predict(X_test)
```

## 📊 Results

The model achieved impressive performance metrics:

| Metric | Score | Baseline study Score |
|--------|-------| -------|
| Accuracy | **98.37** | 96.91
| Recall | 98.12 | - |
| F1 Score | 97.69 | - |
| ROC AUC | 98.31 | - |

These results demonstrate:
- High recall ensures minimal fake articles misclassified as real
- Strong F1 score indicates a balanced approach between precision and recall
- High ROC AUC shows the model effectively distinguishes between real and fake news across different classification thresholds

## 🚀 Improvements Over Prior Work

This work enhances previous approaches [(Patil et al., 2020)](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.685730/full) through:

1. **Binary Classification Focus**: Targeted feature engineering for the specific real vs. fake classification task
2. **URL Feature Integration**: Leveraging source credibility signals from website and image URLs
3. **Dual TF-IDF Vectorization**: Separate vectorizers for content and source data to capture unique characteristics
4. **Language-Specific Processing**: Using spaCy for improved semantic understanding compared to NLTK
5. **SMOTE Oversampling**: Maintaining minority class integrity instead of undersampling, preserving valuable data

## 🔗 References

| Key Concepts | References | 
| ----------- | ----------- | 
| Baseline study | Patil, Vikas, and S. B. Patil. [Source Based Fake News Classification using Machine Learning.](http://www.ijirset.com/upload/2020/june/115_4_Source.PDF) International Journal of Innovative Research in Science, Engineering and Technology (IJIRSET), vol. 7, no. 6, June 2020, pp. 121-124. |
| URL-based features | Mazzeo, V., Rapisarda, A., & Giuffrida, G. (2021). [Detection of fake news on COVID-19 on web search engines.](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.685730/full) Frontiers in Physics, 9. |

---

### 👨‍💻 Contact

For questions or collaboration opportunities, please open an issue or contact [csr.sanjanar@gmail.com](mailto:csr.sanjanar@gmail.com).
