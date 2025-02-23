# Sentiment Analysis on Twitter Data using LSTM

## 📌 Project Overview  
This project implements a **sentiment analysis model** using a **Bidirectional LSTM (Long Short-Term Memory) network** on Twitter data. The model is trained to classify tweets into **four sentiment categories**.  

---

## 🚀 Tech Stack  
- **Programming Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Machine Learning**: LSTM, Tokenization, Embedding, Label Encoding  
- **Data Processing**: Text Cleaning, Padding Sequences  

---

## 📊 Dataset  
- **Training Data**: `twitter_training.csv`  
- **Validation Data**: `twitter_validation.csv`  
- The dataset includes tweets labeled with **four sentiment categories**:  
  - **Negative**  
  - **Positive**  
  - **Neutral**  
  - **Irrelevant**  

---

## ✨ Features Implemented  
✅ Preprocessing of text data (cleaning, tokenization, and padding)  
✅ Implementation of an **LSTM-based deep learning model**  
✅ Model evaluation using accuracy and loss metrics  
✅ Visualization of training and validation accuracy/loss  

---

## 🏗️ Model Architecture  
- **Embedding Layer**  
- **Bidirectional LSTM Layers** with dropout and batch normalization  
- **Fully Connected Dense Layers**  
- **Output Layer** with softmax activation for multi-class classification  

---

## 📈 Results  
📌 **Training Accuracy**: ~91.98%  
📌 **Validation Accuracy**: ~92.29%  
📌 **Test Loss**: ~0.3226  

---

## 🛠️ Installation & Usage  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Vaishnavi2101/Twitter-Sentiment-Analysis-With-LSTM.git
cd Twitter-Sentiment-Analysis-With-LSTM
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Training Script**  
```bash
python train.py
```

### **4️⃣ Evaluate the Model**  
```bash
python evaluate.py
```


