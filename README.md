# Fake News Detector

AI-powered fake news detection using fine-tuned BERT models with a clean Streamlit interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![BERT](https://img.shields.io/badge/model-BERT-green.svg)

## Features

- **Fine-tuned BERT Model** for accurate fake news classification
- **Real-time Analysis** with confidence scores
- **Clean Web Interface** inspired by ChatGPT design
- **Visual Analytics** with interactive charts
- **Sample Articles** for testing

## Quick Start

```bash
# Clone repository
git clone https://github.com/dawoodarain3/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python train_model.py

# Run web app
streamlit run app.py
```

## Model Training

The training script (`train_model.py`) includes:
- Custom BERT fine-tuning for fake news detection
- Sample dataset with real and fake news examples
- Training visualization and evaluation metrics
- Model saving to `./fake_news_model/`

**Training Results:**
- Separate plots for training loss, validation loss, and accuracy
- Confusion matrix with proper styling
- Classification report with precision/recall

## Web Application

The Streamlit app (`app.py`) features:
- **Model Selection**: Choose between F-1 (fine-tuned) or demo model
- **Text Input**: Paste articles for analysis
- **Sample Articles**: Pre-loaded examples
- **Real-time Results**: Instant predictions with confidence scores
- **Visual Charts**: Interactive probability distributions

## Usage

1. **Load Model**: Select F-1 model for best accuracy
2. **Input Text**: Enter news article or use samples
3. **Analyze**: Get prediction with confidence score
4. **Interpret**: Review AI insights and recommendations

## Model Performance

- **F-1 Model**: Fine-tuned BERT optimized for fake news detection
- **Demo Model**: Base BERT (not recommended for production)
- **Confidence Scoring**: Reliability indicator for predictions
- **Visual Feedback**: Color-coded results (Green=Real, Red=Fake)

## Project Structure

```
fake-news-detector/
├── app.py              # Streamlit web application
├── train_model.py      # Model training script
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── fake_news_model/   # Trained model files
```

## Important Notes

⚠️ **Educational Purpose Only** - Always verify through multiple sources  
⚠️ **Not 100% Accurate** - AI predictions may have errors  
⚠️ **Human Verification** - Use critical thinking and fact-checking  

## Contact

**Dawood Ahmed**
- 🔗 [LinkedIn](https://www.linkedin.com/in/dawood-ahmed-84776017b/)
- 📧 [dawoodarain025@gmail.com](mailto:dawoodarain025@gmail.com)
- 🐙 [GitHub](https://github.com/dawoodarain3)

---

⭐ **Star this project if you found it helpful!** Example
```
Input: "Scientists discover miracle pill for instant weight loss..."
Output: ⚠️ SUSPICIOUS NEWS (Confidence: 92.1%)
```

## 🔧 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Set GPU device (optional)
- `TOKENIZERS_PARALLELISM`: Set to false to avoid warnings

### Model Settings
- **Max Tokens**: 256 (configurable in code)
- **Device**: Auto-detection (CUDA/CPU)
- **Batch Size**: 1 (real-time prediction)

## 🚨 Important Disclaimers

⚠️ **Educational Purpose Only**: This tool is designed for educational and research purposes.

⚠️ **Not 100% Accurate**: AI predictions are not infallible. Always verify through multiple reliable sources.

⚠️ **Human Verification Required**: Use critical thinking and cross-reference with trusted news sources.

⚠️ **Bias Considerations**: Model performance may vary across different topics, sources, and writing styles.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- Model improvements
- Documentation updates
- Performance optimizations

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced model architectures (RoBERTa, DistilBERT)
- [ ] Real-time URL analysis
- [ ] Batch processing capabilities
- [ ] API endpoint for integration
- [ ] Mobile-responsive design improvements
- [ ] Export analysis reports

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**
```bash
# Ensure model files exist in ./fake_news_model/
# Check file permissions and directory structure
```

**CUDA Out of Memory**
```bash
# Reduce max_length or use CPU instead
device = torch.device('cpu')
```

**Slow Performance**
```bash
# Enable GPU if available
# Reduce input text length
# Close unnecessary applications
```


## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **Streamlit** for the amazing web framework
- **PyTorch** team for the deep learning framework
- **BERT** authors for the revolutionary architecture

## 📞 Contact & Support

**Developer**: Dawood Ahmed

- 🔗 **LinkedIn**: [linkedin.com/in/dawood-ahmed-84776017b](https://www.linkedin.com/in/dawood-ahmed-84776017b/)
- 📧 **Email**: [dawoodarain025@gmail.com](mailto:dawoodarain025@gmail.com)
- 🐙 **GitHub**: [github.com/dawoodarain3](https://github.com/dawoodarain3)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ by [Dawood Ahmed](https://github.com/dawoodarain3)