# 🎬 IMDB Sentiment Analysis with LSTM

Deep Learning project implementing a **many-to-one LSTM** architecture for sentiment classification of movie reviews.

## 📊 Dataset
- **Source:** [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews
- **Classes:** Binary (Positive/Negative)

## 🧠 Architecture
- Many-to-one LSTM model
- Cell state mechanism (forget gate, input gate, output gate)
- Long-term dependency learning for sentiment nuances

## 🛠️ Tech Stack
- Python 3.12
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn

## 📁 Project Structure
```
imdb-sentiment-lstm/
├── data/              # Dataset
├── models/            # Saved models
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
└── main.py           # Entry point
```

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Optional: Virtual Environment
If you don't have a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

## 📚 University Project
Created as part of Deep Learning coursework.

## 📄 License
MIT License