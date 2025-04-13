*Fake News Detection Using Content and Source Analysis*

This project explores the detection of fake news by analyzing both the content and source credibility of news articles, achieving state-of-the-art performance on a source-based fake news [dataset](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification )
- Using an `AdaBoost classifier` trained on features extracted from text and `URLs` (Based on literature [below](https://github.com/csanjanar/Source-Based-Fake-News-Classification/new/main?filename=README.md#-references))

# Key Contributions 

## Enhanced Feature Engineering
The project goes beyond basic content analysis by extracting richer features from news articles and their sources. This includes lexical analysis of the text and cleaning and incorporating URL information

### 📌 Data Preprocessing

**Text Cleaning** | *URL-based features*

Other than cleaning/normalizing text data with lowercase conversion, removing HTML tags and unwanted characters and numbers, `URLs` were cleaned and tokenized to extract relevant source information
* Curated to retain tokens that represent aspects of the source to incorporate suggestive [URL-based features](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.685730/full):
  * `article/image title` found in the *image/link URL*,
  > *"URLs were found to contain several suggestive word tokens"*
  * `domain` .com, .org, etc.
  > *"The use of dots for adding an extension (i.e., .co) could suggest a fake website"*
  > *"the proportion of http/s did not provide relevant information, as https secured protocol now is commonly used"*

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
  
**Tokenization and Lemmatization:** 

Applied to the content to normalize words to their base form using `spaCy` *(to maintain `linguistic context`; across `different languages`)*
  * The code handles both English and German text, using language-specific models from spaCy.
  * Lemmatization was applied to the 'content' (article text), while the 'source' (metadata) was only tokenized to preserve proper nouns. 

    ```python
      # Load English and German models
      nlp_en = spacy.load("en_core_web_sm")
      nlp_de = spacy.load("de_core_news_sm")
  
      def tokenize_and_lemmatize(content, lang='en'):
      # Select the correct NLP model based on language
      nlp = nlp_de if lang == 'de' else nlp_en
      # Process the text
      doc = nlp(content)
      # Extract lemmas only
      lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
      return lemmas
  
      def tokenize_source(source):
        # Process the source string using English NLP model for simplicity
        doc = nlp_en(source)
        # Return tokens
        tokens = [token.text for token in doc if not token.is_punct]
        return tokens
      ```
**Feature Extraction:** 
`TF-IDF` Vectorization was used to convert text data into numerical features, giving weight to important words
  * Separate vectorizers were used for 'content' and 'source' to customize the vectorization process for each - for richer features
  * TF-IDF was chosen over word embeddings (Word2Vec, GloVe) because:
    * TF-IDF performed better in the [baseline study](http://www.ijirset.com/upload/2020/june/115_4_Source.PDF) and is more suitable for incorporating URL-based features, which do not have strong semantic relationships.
    * TF-IDF also excels when specific keywords have strong classification power.  

      ```python
      
        # Convert lists of lemmas and tokens to single strings - as required by tfidf vectorizer
        df_encoded['content_str'] = df_encoded['content_lemmas'].apply(lambda x: ' '.join(x))
        df_encoded['source_str'] = df_encoded['source_tokenized'].apply(lambda x: ' '.join(x))
    
        # Initialize TF-IDF Vectorizers
        tfidf_vectorizer_content = TfidfVectorizer()
        tfidf_vectorizer_source = TfidfVectorizer()
        
        # Fit and Transform Data
        content_tfidf = tfidf_vectorizer_content.fit_transform(df_encoded['content_str'])
        source_tfidf = tfidf_vectorizer_source.fit_transform(df_encoded['source_str'])
        ```
  
---

## Class Imbalance Handling
Addressing the imbalance between "Real" and "Fake" news articles in the dataset, to avoid biasing the model towards the majority class as shown below:

![image](https://github.com/user-attachments/assets/bdd19097-e91a-49ad-a0f0-1448094e19f6) ![image](https://github.com/user-attachments/assets/13d9c96e-b247-4c01-8406-a5902d3a5896)


### 📌 Data Balancing: 
**Synthetic Minority Over-sampling (SMOTE)** was used to `oversample` the minority class to address class imbalance. 
* This approach was chosen based on [this study](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.685730/full) which states, "with the under-sampling technique, where instances from the majority class were removed, the score of the classifier models was very poor compared to the over-sampling technique".

  ```python
  
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)
  ```
  ```
  Before SMOTE: label_encoded
  0    890
  1    542
  Name: count, dtype: int64
  After SMOTE: label_encoded
  1    890
  0    890
  Name: count, dtype: int64
  ```
---

## Binary Classification Focus
Unlike some previous work that classified news into multiple categories, this project focuses on binary classification (Real or Fake) to create a more generalizable and adaptable model.

### 📍 Training

**Ensemble Learning** 
*(AdaBoost classifier with Decision Trees as base estimator)*
  * The choice of this model was primarily motivated by its superior performance in the [baseline study](http://www.ijirset.com/upload/2020/june/115_4_Source.PDF), Noting that:
    * `AdaBoost` is robust to noisy data, effective with high-dimensional data, and focuses on hard-to-classify instances.
    * `Decision Trees` were chosen as the base estimator for their interpretability and efficiency. 
    * `Parameters` used were derived from a [previous study](https://www.kaggle.com/code/ruchi798/how-do-you-recognize-fake-news#Label-vs-Type) that also used the same dataset.

      ```python
        # Initialize the AdaBoost classifier
        # With decision tree as the base estimator
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
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

### 📍 Evaluation

The models were evaluated using the following metrics to provide a more holistic view of the model's effectiveness than accuracy alone, especially in the context of fake news detection where minimizing false negatives (failing to identify fake news) is crucial.

*Recall:* **0.98**13953488372092 
* High recall ensures that the system minimises the number of fake articles that are misclassified as real, addressing the critical need for comprehensive detection.

*F1 score:* **0.97**68518518518519
* Ensures that the model does not overly favour recall over precision or vice versa, promoting a balanced approach to both catching fake news and maintaining credibility.

*ROC AUC:* **0.98**31788774261235
* The ROC AUC provides a comprehensive measure of a model’s effectiveness across all possible classification thresholds.
* In fake news detection, a high AUC value indicates that the model accurately distinguishes between real and fake news, allowing flexibility in adjusting the threshold to meet different operational needs without sacrificing overall performance.

### 📝 References
| Key Concepts | References | 
| ----------- | ----------- | 
| Baseline study | Patil, Vikas, and S. B. Patil. "[Source Based Fake News Classification using Machine Learning.](http://www.ijirset.com/upload/2020/june/115_4_Source.PDF)" International Journal of Innovative Research in Science, Engineering and Technology (IJIRSET), vol. 7, no. 6, June 2020, pp. 121-124. |
| URL-based features | Mazzeo, V., Rapisarda, A., & Giuffrida, G. (2021). [Detection of fake news on COVID-19 on web search engines.](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.685730/full) Frontiers in Physics, 9. |
