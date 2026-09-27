# 📧 Email Spam Classifier — Naive Bayes / Text Classification

A complete, end-to-end Naive Bayes text-classification project that detects spam emails using classic NLP preprocessing, Bag-of-Words / TF-IDF vectorization, hyperparameter tuning, feature-selection based vocabulary
pruning, and word-level model explainability.

> **Project 04** 

---

## 📁 Repository Contents

| File | Description |
|---|---|
| `spam_classifier.ipynb` | The full, already-executed Jupyter notebook — every code cell has a "what we're doing / expected output" markdown cell before it, and a "what happened" markdown cell after it. |
| `combined_data.csv` | The dataset — `label` (1 = spam, 0 = ham) and `text` (raw email content). |
| `README.md` | This file. |

---

## 📊 Dataset

**Spam Email Classification Dataset** — a combined corpus built from the **2007 TREC Public Spam Corpus** and
the **Enron-Spam Dataset**.

- **Source:** [Kaggle — Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)
- **Rows:** 83,448 emails
- **Columns:** `text` (subject + body, headers/formatting artifacts included), `label` (`1` = spam, `0` = ham)
- **Class balance:** 43,910 spam (52.6%) vs 39,538 ham (47.4%) — mildly imbalanced
- **Missing values:** none

The dataset is intentionally messy — near-duplicate spam templates, leftover HTML tags, email headers, and
forwarded-message chains — which forces a real text-cleaning pipeline instead of a "plug numbers into a model"
exercise.

---

## 🧭 Project Workflow

The notebook works through 8 stages, matching the assignment's task breakdown:

| # | Stage | What it covers |
|---|---|---|
| 1 | **EDA + Preprocessing** | Class balance, email-length distribution, duplicate check, a full text-cleaning function (lowercase → strip HTML/URLs → remove punctuation/numbers → tokenize → remove stopwords), stratified 80/20 train-test split |
| 2 | **Baseline Model** | An intentionally *unconstrained* `CountVectorizer` + `MultinomialNB` to expose the train/test overfitting gap |
| 3 | **Vectorization Comparison** | `CountVectorizer` (Bag-of-Words) vs `TfidfVectorizer`, same vocabulary size, same classifier — plus their respective top predictive words |
| 4 | **Hyperparameter Tuning** | `GridSearchCV` over `max_features`, `ngram_range`, and `alpha` (Laplace smoothing) |
| 5 | **Vocabulary Pruning** | `SelectKBest(chi2)` to rank words by informativeness; test accuracy plotted against vocabulary size *k* |
| 6 | **Feature Importance** | Top 18 most "spammy" and most "hammy" words, ranked by `feature_log_prob_` log-ratio |
| 7 | **Explainability** | One test email decomposed word-by-word into log-odds contributions, reconstructing the model's decision by hand |
| 8 | **Bonus: Obfuscation Test** | Lightly obfuscated spam emails (`free` → `fr33`, stray symbols) re-run through the model to test robustness |

### ⚙️ Code design

The notebook is built around one shared **config block** and a small set of **reusable helper functions**
(`compute_metrics`, `print_metrics`, `spam_ham_log_ratio`, `top_words`) defined once near the top, so:
- every magic number (test size, vocabulary caps, sample size, top-N words) lives in a single `CONFIG` dict
- metric calculation/printing and word-ranking logic are never duplicated across sections
- Sections 5–8 **reuse one shared vectorizer and vectorized matrix** instead of re-vectorizing the full training
  set four separate times, keeping the notebook fast to (re-)run

---

## 📈 Key Results

| Model | Vocabulary | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (unconstrained CountVectorizer) | 243,971 words | 97.45% | 98.92% | 96.20% | 97.54% |
| CountVectorizer (capped, 5,000 words) | 5,000 words | 95.45% | 97.63% | 93.62% | 95.58% |
| TfidfVectorizer (capped, 5,000 words) | 5,000 words | 96.45% | 97.57% | 95.64% | 96.60% |
| **Tuned model (GridSearchCV best params)** | Full vocab + bigrams, `alpha=0.01` | **98.23%** | **99.48%** | 97.14% | 98.29% |

**Best hyperparameters found:** `max_features=None`, `ngram_range=(1,2)`, `alpha=0.01`

**Vocabulary pruning:** accuracy climbs steeply to ~91% by 1,000 words, then rises slowly to 96.70% at the full
20,000-word vocabulary — no sharp elbow, but ~2,000–3,000 words already captures most of the achievable accuracy
at a fraction of the vocabulary size.

**Robustness:** 1 out of 8 correctly-flagged spam emails flipped to "ham" after simple character-level
obfuscation (`o`→`0`, `e`→`3`, etc.) — a known, real limitation of bag-of-words/TF-IDF models.

---

## 🔍 Notable Insights

- **Overfitting is real but modest:** an unconstrained 243,971-word vocabulary produces only a ~0.67-point
  train/test accuracy gap — Naive Bayes is naturally fairly resistant to overfitting compared to models like
  decision trees.
- **TF-IDF beats raw counts** at the same vocabulary size, and surfaces more recognizable spam vocabulary
  (`viagra`, `cialis`, `pharmacy`) versus CountVectorizer's rarer, campaign-specific tokens.
- **A bigger vocabulary tuned correctly wins:** the best model uses the *full* vocabulary plus bigrams with very
  light smoothing — shrinking the vocabulary is not always the right anti-overfitting move.
- **A dataset-specific caveat:** the most "ham-indicating" words are largely **Enron-specific names and
  technical/mailing-list jargon** (`vince`, `kaminski`, `samba`, `winbindd`) rather than generic non-spam
  language — a reminder that a model trained on this dataset may not generalize perfectly to a different
  organization's inbox.

---

## 🛠️ Setup & How to Run

### Requirements
```
python >= 3.9
pandas
numpy
matplotlib
scikit-learn
jupyter
```

### Install
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Run
```bash
jupyter notebook spam_classifier.ipynb
```
Make sure `combined_data.csv` is in the same folder as the notebook before running, or update the `pd.read_csv(...)`
path in Section 1.1.

> The notebook runs fully offline — text cleaning uses scikit-learn's built-in English stopword list, so no
> NLTK data download is required.

---

## 🚧 Limitations & Future Improvements

- **Exact-duplicate check only:** near-duplicate spam templates (same message, slightly reworded) are not
  caught — a fuzzy-matching / minhash-based deduplication step could tighten this up.
- **Word-level tokens are exploitable:** the obfuscation test shows a spammer can dodge detection with simple
  character substitutions. **Character n-grams** instead of word n-grams would likely be more robust.
- **Corpus-specific ham signal:** since the ham class comes entirely from one company's mailbox (Enron), some of
  the model's "ham" signal is arguably learning "this looks like this company's mail" rather than "this is not
  spam" in general. Testing on a different, independent ham source would validate real-world generalization.
- **Hyperparameter search used a training subsample** (15,000 emails) for speed; a full-data grid search on more
  compute would likely refine the best parameters slightly further.

---

## ✍️ Author

**Joytu Mallik**

