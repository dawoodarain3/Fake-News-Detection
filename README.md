
# Fake News Detector

AI fake news detection using a fine-tuned BERT model with a clean Streamlit interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![BERT](https://img.shields.io/badge/model-BERT-green.svg)

---

## Features

- Fine-tuned BERT model for high-accuracy fake news classification  
- Real-time analysis with confidence scoring  
- Modern web UI inspired by ChatGPT  
- Interactive visual analytics  
- Preloaded sample articles for testing  

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/dawoodarain3/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python train_model.py

# Launch the app
streamlit run app.py
```

---

## Download My Custom Model

You can download the **fine-tuned BERT model (RB-1)** from Kaggle and skip training from scratch.

### 🔗 [Download Custom Model from Kaggle](https://www.kaggle.com/models/dawoodarain/fake_news_model)

> **Note:** Make sure you have a Kaggle account and are logged in to access the dataset.

### How to Use the Model

1. Download the ZIP file containing the custom model from Kaggle  
2. Extract the ZIP contents into the `fake_news_model/` directory (create it if it doesn’t exist):

```bash
# Example shell commands (Linux/macOS/WSL)
mkdir -p fake_news_model
unzip /path/to/downloaded_model.zip -d fake_news_model/


---

## Model Training

The `train_model.py` script includes:
- Fine-tuning BERT on real/fake news data
- Visualization of training and validation performance
- Model saving to `./fake_news_model/`

Outputs include:
- Training and validation loss plots  
- Accuracy graphs  
- Confusion matrix  
- Classification report  

---

## Web Application Overview

The `app.py` Streamlit app includes:
- Model selection (RB-1 or demo)
- Input field for text or use of sample articles
- Instant predictions with confidence scoring
- Visual display of probability distribution

---

## Usage

1. Select the model (RB-1 for best performance)  
2. Input your news article or choose a sample  
3. Click analyze to get the result  
4. Review the prediction and confidence score  

---

## Model Performance

| Model      | Description                         | Recommended |
|------------|-------------------------------------|-------------|
| RB-1       | Fine-tuned BERT on fake news data   | Yes         |
| Demo       | Base BERT without tuning            | No          |

- Confidence score indicates prediction certainty  
- Color-coded output: Green for Real, Red for Fake  

Example:  
```
Input: "Scientists discover miracle pill for instant weight loss..."
Output: SUSPICIOUS NEWS (Confidence: 92.1%)
```

---

## Project Structure

```
fake-news-detector/
├── app.py              # Streamlit web application
├── train_model.py      # Model training script
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── fake_news_model/    # Saved model files
```

---

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Set GPU device (optional)  
- `TOKENIZERS_PARALLELISM`: Set to false to suppress warnings  

### Model Settings
- Max tokens: 256  
- Device: Auto-detection (CPU or GPU)  
- Batch size: 1 (for real-time inference)  

---

## Troubleshooting

**Model Not Loading**  
- Ensure `./fake_news_model/` directory and model files exist  
- Check file permissions and directory structure  

**CUDA Memory Error**  
- Switch to CPU mode by setting `device = torch.device('cpu')`  
- Reduce input length or batch size  

**Slow Performance**  
- Enable GPU if available  
- Reduce input text size  
- Close unnecessary background applications  

---

## Future Enhancements

- Multi-language support  
- Integration with advanced models (RoBERTa, DistilBERT)  
- Real-time URL analysis  
- Batch processing  
- REST API support  
- Mobile responsiveness  
- Report exporting  

---

## Contributing

Contributions are welcome.

Steps:
1. Fork the repository  
2. Create a new branch: `git checkout -b feature-name`  
3. Make your changes and test them  
4. Submit a pull request with a detailed description  

Areas to contribute:
- Feature enhancements  
- Bug fixes  
- Model improvements  
- Documentation updates  
- Performance optimizations  

---

## Acknowledgments

- Hugging Face for the Transformers library  
- Streamlit for the web interface  
- PyTorch for deep learning infrastructure  
- BERT authors for the model architecture  

---

## Contact

**Developer:** Dawood Ahmed  
- LinkedIn: [linkedin.com/in/dawood-ahmed-84776017b](https://www.linkedin.com/in/dawood-ahmed-84776017b/)  
- GitHub: [github.com/dawoodarain3](https://github.com/dawoodarain3)  
- Email: [dawoodarain025@gmail.com](mailto:dawoodarain025@gmail.com)  

---

If you found this project helpful, please consider starring the repository.

Made by Dawood Ahmed
