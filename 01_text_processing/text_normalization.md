# Text Normalization in NLP

Text normalization is a **core preprocessing step** in Natural Language Processing (NLP). It converts raw, noisy text into a **standardized and consistent form**, improving downstream tasks like tokenization, embeddings, and model performance.


## 1. What is Text Normalization?

Text normalization is the process of **transforming text into a canonical form** by reducing variability caused by casing, punctuation, spelling variations, and linguistic inflections.

> Goal: *Different surface forms → same meaning representation*

Example:

```
"Running", "runs", "ran" → "run"
```

## 2. Why Text Normalization is Important

* Reduces vocabulary size
* Improves statistical efficiency
* Helps models generalize better
* Essential for NLP, LLMs, and search systems


## 3. Common Text Normalization Techniques

### 3.1 Lowercasing

Converts all characters to lowercase.

```python
text = "Natural Language Processing"
text.lower()
```

Pros:

* Reduces vocabulary size

Cons:

* Loses proper noun information ("US" vs "us")


### 3.2 Removing Punctuation

```python
import re
text = "Hello!!! NLP is awesome."
re.sub(r"[^\w\s]", "", text)
```

### 3.3 Removing Numbers

```python
text = "AI will dominate by 2026"
re.sub(r"\d+", "", text)
```


### 3.4 Removing Stopwords

Stopwords are common words that add little semantic meaning.

Examples:

* is, am, are, the, a, an

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words = word_tokenize("NLP is very powerful")
[w for w in words if w not in stop_words]
```


## 4. Stemming

Stemming reduces words to their **root form** by applying heuristic rules.

Example:

```
playing → play
studies → studi
```

### 4.1 Porter Stemmer

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmer.stem("running")
```

Pros:

* Fast
* Simple

Cons:

* Can produce non-dictionary words

## 5. Lemmatization

Lemmatization reduces words to their **dictionary base form (lemma)** using linguistic knowledge.

Example:

```
better → good
running → run
```

### 5.1 WordNet Lemmatizer

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("running", pos='v')
```

Pros:

* Linguistically correct

Cons:

* Slower than stemming
* Requires POS tagging


## 6. Stemming vs Lemmatization

| Feature      | Stemming  | Lemmatization   |
| ------------ | --------- | --------------- |
| Speed        | Fast      | Slower          |
| Accuracy     | Low       | High            |
| Output       | Root form | Dictionary word |
| POS required | No        | Yes             |


## 7. Text Normalization Pipeline

```python
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text
```


## 8. When NOT to Normalize Aggressively

* Named Entity Recognition (NER)
* Sentiment analysis (case/emojis matter)
* Legal or medical text


## 9. Summary

* Text normalization standardizes text
* Includes lowercasing, stopword removal, stemming, lemmatization
* Improves model performance and efficiency
* Choice depends on task and model


## Next Topics

* text_representation

**Author:** Monower Hossen <br>
**Domain:** ML/AI | NLP | LLM | Generative AI
