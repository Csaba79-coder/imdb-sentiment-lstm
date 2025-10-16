# 🎬 IMDB Sentiment Analysis with LSTM

Deep Learning project implementing a **many-to-one LSTM** architecture for sentiment classification of movie reviews.

## 📊 Dataset
- **Source:** [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Original Size:** 50,000 reviews
- **Cleaned Size:** 49,578 reviews (422 duplicates removed)
- **Classes:** Binary (Positive/Negative)
- **Balance:** 1.01:1 (50.18% positive, 49.82% negative) ✅ Perfectly balanced!

## 📊 Dataset Setup

### Download the Dataset
1. Visit [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Download `IMDB Dataset.csv`
3. Place it in `data/` folder as `imdb_dataset.csv`

### Or use Kaggle API (automated):
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/
mv data/IMDB\ Dataset.csv data/imdb_dataset.csv
```

## 🧠 Architecture
- Many-to-one LSTM model
- Cell state mechanism (forget gate, input gate, output gate)
- Long-term dependency learning for sentiment nuances

## 🛠️ Tech Stack
- **Python:** 3.12 (stable version)
- **Deep Learning:** TensorFlow 2.20.0
- **Data Processing:** NumPy 1.26.x, Pandas 2.3.3
- **Visualization:** Matplotlib 3.10.7, Seaborn 0.13.2, WordCloud 1.9.4
- **NLP:** NLTK 3.9.2
- **ML Utils:** scikit-learn 1.7.2

## 📁 Project Structure
```
imdb-sentiment-lstm/
├── .venv/                            # Virtual environment
├── data/
│   ├── imdb_dataset.csv              # Original dataset (50K reviews)
│   ├── imdb_dataset_formatted.csv    # HTML tags removed
│   └── imdb_dataset_cleaned.csv      # Final cleaned (49,578 reviews)
├── models/                           # Saved trained models
├── notebooks/                        # Jupyter notebooks for experiments
├── visualizations/
│   ├── eda/                          # Exploratory Data Analysis plots
│   ├── preprocessing/                # Preprocessing visualizations
│   └── training/                     # Training history plots
├── src/
│   ├── __init__.py
│   ├── check_versions.py             # PyPI version checker
│   ├── config.py                     # Configuration & hyperparameters
│   ├── data_clean.py                 # Data cleaning & EDA
│   ├── data_inspect.py               # Initial data inspection
│   └── data_loader.py                # Data loading utilities
├── .gitignore
├── LICENSE
├── main.py                           # Main entry point
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd imdb-sentiment-lstm
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional: Update pip
If prompted to update pip:
```bash
python -m pip install --upgrade pip
```

## 📋 Data Pipeline

### ✅ Step 1: Data Inspection
Explore the raw dataset structure and basic statistics.
```bash
python src/data_inspect.py
```

**Output:**
- Dataset info (50,000 rows, 2 columns)
- First 5 samples
- Sentiment distribution
- HTML tag detection

---

### ✅ Step 2: Data Formatting
Remove HTML tags and format text for analysis.
```bash
python src/data_format.py
```

**Output:**
- Cleaned reviews (HTML tags removed)
- Saved to `data/imdb_dataset_formatted.csv`

---

### ✅ Step 3: Data Cleaning & EDA
Comprehensive data cleaning and exploratory analysis.
```bash
python src/data_clean.py
```

**What it does:**
- ✅ **Missing values check:** 0 missing values found
- ✅ **Duplicate removal:** 422 duplicates removed (0.84%)
- ✅ **Text length analysis:** Character & word counts
- ✅ **Outlier detection:** IQR method (7.39% outliers kept)
- ✅ **Descriptive statistics:** Mean, median, std, min, max
- ✅ **7 Visualizations created:**
  - Sentiment distribution (bar chart)
  - Text length histogram (word & character count)
  - Text length boxplot (by sentiment)
  - Word clouds (positive & negative)
  - Top 20 frequent words (positive & negative)

**Output:**
- `data/imdb_dataset_cleaned.csv` (49,578 reviews)
- 7 PNG visualizations in `visualizations/eda/`

**Key Statistics:**
```
Total Reviews:     49,578
Positive:          24,882 (50.18%)
Negative:          24,696 (49.82%)
Avg Word Count:    229 words
Median Word Count: 172 words
```

---

### 🔜 Step 4: Data Preprocessing
- Tokenization
- Sequence padding
- Train/test split
- Vocabulary creation

---

### 🔜 Step 5: Model Training
- LSTM architecture definition
- Model compilation
- Training with validation
- Performance evaluation

---

## 📊 Exploratory Data Analysis Results

After cleaning, our dataset shows excellent characteristics for training:

- **Perfect Balance:** 50.18% positive vs 49.82% negative (no resampling needed!)
- **Good Text Length Distribution:** Average 229 words, suitable for LSTM
- **Minimal Duplicates:** Only 0.84% removed
- **No Missing Data:** 100% complete dataset
- **Outliers Kept:** 7.39% long/short reviews retained (may contain valuable sentiment information)

Check the visualizations in `visualizations/eda/` for detailed insights! 📈

---

## 🎓 University Project
Created as part of Deep Learning coursework at University of Pannonia.

## 👨‍💻 Author
Developed with focus on understanding LSTM mechanisms and practical NLP implementation.

## 📄 License
MIT License

---

## 🐛 Troubleshooting

### Issue: Import errors after installing dependencies
**Solution:** Make sure you're in the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Issue: TensorFlow compatibility warnings
**Solution:** We use NumPy 1.26.x (not 2.x) for TensorFlow compatibility. This is intentional and stable.

### Issue: WordCloud not found
**Solution:** Install separately if needed:
```bash
pip install wordcloud==1.9.4
```

### Issue: Module not found errors
**Solution:** Run scripts from project root:
```bash
python src/data_clean.py  # ✅ Correct
cd src && python data_clean.py  # ❌ Wrong
```
