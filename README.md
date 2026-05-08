# Metaphor Detection Using Machine Learning & Semi-Supervised Learning

## Project Overview

This project focuses on detecting metaphorical expressions in Nigerian poetic text using both supervised and semi-supervised machine learning techniques. The system combines Natural Language Processing (NLP), word embeddings, and classification algorithms to identify whether a sentence contains a metaphor.

The project explores how linguistic features such as semantic mappings, conceptual relationships, and contextual word embeddings can improve metaphor classification performance.

A major highlight of this work is the comparison between:

* Traditional supervised learning models
* Semi-supervised learning (SSL) approaches using pseudo-labeling on unlabeled the literary data

The study demonstrates how unlabeled data can help improve metaphor detection performance when labeled data is limited.

---

# Problem Statement

Metaphors are widely used in literature and human communication, but detecting them computationally is challenging because figurative meanings often differ from literal interpretations.

Traditional rule-based approaches struggle to generalize across different writing styles and contexts.

This project aims to:

* Automatically classify metaphorical and non-metaphorical text
* Extract meaningful semantic relationships from text
* Compare supervised learning with semi-supervised learning approaches
* Investigate how unlabeled data can contribute to metaphor detection

---

# Dataset Description

The project uses:

* **Labeled dataset** → 509 manually annotated literary excerpts
* **Unlabeled dataset** → 14,994 literary excerpts

### Labeled Dataset Features
![Labelled Head](images/labelled_head.PNG)
| Column   | Description                                   |
| -------- | --------------------------------------------- |
| Excerpt  | Literary sentence or phrase                   |
| HPSM     | High-level metaphor semantic mapping          |
| LPSM     | Low-level semantic mapping                    |
| Concepts | Extracted metaphor-related concepts           |
| Label    | Target class (1 = metaphor, 0 = non-metaphor) |


### Class Distribution

* Metaphor - 255
* Non-metaphor - 254
  
The dataset is balanced, which helps reduce model bias during training.

---

# Project Workflow

# Step 1: Import Libraries

The project begins by importing NLP, machine learning, and visualization libraries such as NumPy, Pandas, spaCy, Scikit-learn, Seaborn and Matplotlib which was used for Text preprocessing, Vectorization, Model training, Evaluation and Visualization

---

# Step 2: Load spaCy Language Model

The `en_core_web_lg` spaCy model was downloaded and loaded and this model provides functions like Tokenization, Lemmatization, Part-of-speech tagging, Dependency parsing and 300-dimensional word vectors
These embeddings became the foundation for semantic understanding in the project.

---

# Step 3: Text Preprocessing

A custom preprocessing pipeline was created to tokenize text, Convert words to lowercase, remove punctuation, remove extra whitespaces and lemmatization of words. This preprocessing helps standardize textual data before vectorization.
![Labelled processed Head](images/labelled_preprocessed_head.PNG)

---

# Step 4: Feature Engineering

The following text columns were individually preprocessed:

* Excerpt
* HPSM
* LPSM
* Concepts

After preprocessing, all features were concatenated into a single feature column:

```python
Concatenated_Column = Preprocessed_Excerpt + Preprocessed_HPSM + Preprocessed_LPSM + Preprocessed_Concepts
```

This allowed the model to learn from contextual meaning, semantic mappings and extracted metaphor concepts instead of relying only on raw text.

---

# Step 5: Word Embedding Generation

Each concatenated text was transformed into a numerical vector using spaCy embeddings:

Each sentence became a 300-dimensional dense vector representation an this in turn helped captured the semantic similarity, the contextual relationships and metaphorical meaning from the exercept.

---

# Step 6: Train-Test Split

The labeled dataset was split into 80% training data and 20% testing data to ensure unbiased model evaluation.

---

# Step 7: Supervised Learning Models

Several supervised learning algorithms were trained and evaluated. The models used includes:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Logistic Regression
* Decision Tree Classifier

---

# Step 8: Model Evaluation

Models were evaluated using accuracy, precision, recall, F1-score and Confusion Matrix. 
## Supervised Learning Results
![Decision Tree Confusion Matrix](images/DT_matrix.PNG)

The Decision Tree classifier achieved an accuracy of 85% on the test dataset. From the confusion matrix, the model correctly identified 43 metaphorical instances (True Positives) and 44 non-metaphorical instances (True Negatives). However, the model produced 10 False Positives, where literal expressions were incorrectly classified as metaphors, and 5 False Negatives, where actual metaphors were missed.

This result suggests that while the Decision Tree model was able to capture important decision boundaries within the semantic embedding space, it struggled with generalization compared to the other models. The relatively higher number of False Positives indicates that the model occasionally overfitted to certain semantic patterns, causing it to incorrectly label some literal expressions as metaphorical.

![Logistic Regression Confusion Matrix](images/LR_matrix.png)

The Logistic Regression classifier demonstrated strong predictive performance with an overall accuracy of 95%. The confusion matrix shows that the model correctly classified 50 metaphorical expressions (True Positives) and 47 non-metaphorical expressions (True Negatives). Only 3 False Positives and 2 False Negatives were recorded.

These results indicate that Logistic Regression was highly effective at separating metaphorical and non-metaphorical language using the generated semantic embeddings. The low number of classification errors demonstrates that the model generalized well and maintained balanced precision and recall across both classes. This performance highlights the suitability of linear models for NLP classification tasks involving dense vector representations.

![Support Vector Machine Confusion Matrix](images/SVM_matrix.png)

The Support Vector Machine (SVM) classifier achieved the best overall performance among all supervised learning models, reaching an accuracy of 96%. According to the confusion matrix, the model correctly predicted 50 metaphorical instances (True Positives) and 48 non-metaphorical instances (True Negatives). The classifier generated only 3 False Positives and 1 False Negative, indicating exceptionally strong classification capability.

The low error rate demonstrates the effectiveness of SVM in handling high-dimensional semantic embedding data. Its ability to maximize class separation allowed it to distinguish metaphorical language patterns more accurately than the other algorithms. The near-perfect balance between precision and recall confirms that the SVM model provided the most reliable and consistent metaphor detection performance in this study.

![KNN Confusion Matrix](images/KNN_matrix.PNG)

The K-Nearest Neighbors (KNN) classifier achieved an accuracy of 89% on the supervised learning task. The confusion matrix reveals that the model correctly identified 49 metaphorical expressions (True Positives) and 42 non-metaphorical expressions (True Negatives). However, the model recorded 4 False Positives and 7 False Negatives.

The higher number of False Negatives suggests that KNN occasionally failed to detect certain metaphorical expressions, likely due to similarities between metaphorical and literal semantic representations in the vector space. Despite this limitation, the model still performed competitively and demonstrated that distance-based learning methods can effectively classify metaphorical language when supported by high-quality semantic embeddings.

![Decision Tree Result](images/DT_result.PNG)
![Logistic Regression Result](images/LR_result.PNG)
![Support Vector Machine Result](images/SVM_result.PNG)
![KNN Result](images/KNN_result.PNG)

| **Model** | **Precision (Class 0)** | **Recall (Class 0)** | **F1‑Score (Class 0)** | **Precision (Class 1)** | **Recall (Class 1)** | **F1‑Score (Class 1)** | **Accuracy** | **Macro Avg (F1)** | **Weighted Avg (F1)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **[Decision Tree](ca://s?q=Decision_Tree_in_supervised_learning)** | 0.90 | 0.81 | 0.85 | 0.81 | 0.90 | 0.85 | 0.85 | 0.85 | 0.85 |
| **[K‑Nearest Neighbors (KNN)](ca://s?q=KNN_in_supervised_learning)** | 0.88 | 0.92 | 0.90 | 0.91 | 0.86 | 0.88 | 0.89 | 0.89 | 0.89 |
| **[Logistic Regression](ca://s?q=Logistic_Regression_in_supervised_learning)** | 0.96 | 0.94 | 0.95 | 0.94 | 0.96 | 0.95 | 0.95 | 0.95 | 0.95 |
| **[Support Vector Machine (SVM)](ca://s?q=SVM_in_supervised_learning)** | 0.98 | 0.94 | 0.96 | 0.94 | 0.98 | 0.96 | 0.96 | 0.96 | 0.96 |

The following table summarizes the performance of all supervised learning models evaluated in this study. SVM and Logistic Regression achieved the strongest overall results, indicating superior generalization and classification capability for metaphor detection tasks.

Hence, **Support Vector Machine (SVM)** is  the Best Performing Model as it achieved the highest performance with:

* 96% accuracy
* a strong precision and recall
* balanced classification performance

This suggests that linear SVMs are highly effective for metaphor classification using semantic embeddings.

---

# Step 9: Unlabeled Data Processing

The unlabeled dataset contained 14,994 literary excerpts. To make the data useful for SSL, HPSM and LPSM markers were automatically generated. Concepts were extracted using:
  * dependency parsing
  * noun phrase extraction
  * metaphorical pair detection
    
![Unlabelled Head](images/Unlabelled_head.PNG)

---

# Step 10: Concept Extraction Strategy

Several NLP strategies were used to identify metaphorical relationships:

## Extracted Pair Types

* Noun–Noun pairs
* Adjective–Noun pairs
* Adverb–Adjective pairs

These linguistic combinations help capture figurative semantic relationships.

---

# Step 11: Semantic Dissimilarity Detection

Cosine similarity was used to identify semantically distant word pairs.

Low similarity scores often indicate metaphorical relationships because metaphorical terms tend to connect unrelated semantic domains. Hence, this became a key feature engineering step in the project.

---

# Step 12: Semi-Supervised Learning (SSL)

The project implemented pseudo-labeling for SSL.

## SSL Process

1. Train initial model on small labeled subset
2. Predict labels for unlabeled data
3. Select high-confidence predictions
4. Add confident predictions to training set
5. Retrain model iteratively

This process allows the model to learn from unlabeled literary data.

---

# Semi-Supervised Learning Results

| Model               | SSL Accuracy |
| ------------------- | ------------ |
| Logistic Regression | 93%          |
| KNN                 | 86%          |
| SVM                 | 80%          |
| Decision Tree       | 79%          |

---

# Key Findings

## 1. Supervised Learning Outperformed SSL

The fully supervised SVM model achieved the best overall performance.

Reason:

* High-quality labeled data
* Balanced dataset
* Strong semantic embeddings

---

## 2. SSL Still Performed Strongly

The SSL Logistic Regression model achieved:

* 93% accuracy

This demonstrates that unlabeled literary data can significantly improve performance when labeled data is limited.

---

## 3. Semantic Embeddings Were Highly Effective

spaCy’s pretrained embeddings successfully captured:

* contextual meaning
* semantic relationships
* metaphorical patterns

without requiring deep neural networks.

---

# Challenges Encountered

## Convergence Warnings

Logistic Regression produced convergence warnings during SSL training.

Possible improvements would include feature scaling, increasing `max_iter` or dimensionality reduction

---

## Metaphor Complexity

Metaphorical language is highly context-dependent, making some expressions difficult to classify accurately.

---

# Future Improvements

Possible future enhancements include:

* Transformer models (BERT, RoBERTa)
* Contextual embeddings
* Attention mechanisms
* Deep learning architectures
* Larger annotated datasets
* Advanced semantic similarity techniques

---
# Example Results

## Supervised SVM Performance

```text
Accuracy: 96%
Precision: 96%
Recall: 96%
F1-score: 96%
```

## SSL Logistic Regression Performance

```text
Accuracy: 93%
Precision: 93%
Recall: 93%
F1-score: 93%
```

---

# Conclusion

This project demonstrates that machine learning and NLP techniques can effectively detect metaphorical language in literary text.

The combination of:

* semantic embeddings
* feature engineering
* linguistic relationship extraction
* semi-supervised learning

resulted in highly accurate metaphor classification models.

The study also highlights the importance of unlabeled data in NLP tasks where annotated datasets are limited.

---

# Author
Latifah Usaini Bashir


