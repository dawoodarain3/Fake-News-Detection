# Clean Fake News Detector - ChatGPT Style UI
# Run with: streamlit run clean_app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import re
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean ChatGPT-style CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main {
    padding: 1rem 2rem;
    width: 100%;
    max-width: none;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

.css-1y4p8pa {
    background-color: #f8fafc;
}

/* Main content area */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: none;
}

/* Responsive layout */
@media (min-width: 768px) {
    .main {
        padding: 1rem 3rem;
    }
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem 0;
}

.header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #1f2937;
    margin: 0 0 0.5rem 0;
}

.header p {
    font-size: 1rem;
    color: #6b7280;
    margin: 0;
}

/* Input Section */
.input-section {
    background: #f9fafb;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #e5e7eb;
}

/* Model Status */
.status-container {
    margin-bottom: 1.5rem;
}

.status-success {
    background: #dcfce7;
    color: #166534;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px solid #bbf7d0;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-error {
    background: #fef2f2;
    color: #991b1b;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px solid #fecaca;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-warning {
    background: #fefbef;
    color: #92400e;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px solid #fed7aa;
    font-size: 0.875rem;
    font-weight: 500;
}

/* Results */
.result-container {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

.prediction-real {
    background: #dcfce7;
    color: #166534;
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
    font-size: 1.125rem;
    font-weight: 600;
    margin: 1rem 0;
    border: 1px solid #bbf7d0;
}

.prediction-fake {
    background: #fef2f2;
    color: #991b1b;
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
    font-size: 1.125rem;
    font-weight: 600;
    margin: 1rem 0;
    border: 1px solid #fecaca;
}

/* Metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-item {
    text-align: center;
    padding: 1rem;
    background: #f9fafb;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
}

/* Progress bars */
.progress-container {
    background: #f3f4f6;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.progress-real { background: #10b981; }
.progress-fake { background: #ef4444; }
.progress-confidence { background: #3b82f6; }

/* Buttons */
.stButton > button {
    background: #1f2937 !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background-color 0.2s !important;
}

.stButton > button:hover {
    background: #111827 !important;
}

/* Text areas */
.stTextArea > div > div > textarea {
    border-radius: 6px !important;
    border: 1px solid #d1d5db !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Radio buttons */
.stRadio > div {
    flex-direction: row !important;
    gap: 2rem !important;
}

/* Insights */
.insight {
    padding: 1rem;
    border-radius: 6px;
    margin: 1rem 0;
    border-left: 4px solid;
    font-size: 0.875rem;
    line-height: 1.5;
}

.insight-info {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1e40af;
}

.insight-success {
    background: #f0fdf4;
    border-color: #10b981;
    color: #166534;
}

.insight-warning {
    background: #fefbef;
    border-color: #f59e0b;
    color: #92400e;
}

.insight-danger {
    background: #fef2f2;
    border-color: #ef4444;
    color: #991b1b;
}

/* Loading */
.loading {
    text-align: center;
    color: #6b7280;
    padding: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.875rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #e5e7eb;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.max_length = 256
        self.model_loaded = False
        
    def preprocess_text(self, text):
        if not text:
            return ""
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_model(self, model_path=None):
        """Load model without caching to avoid session state issues"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = BertForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                model_type = "Custom Trained Model"
                is_trained = True
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=2
                )
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model_type = "Demo Model (Not Fine-tuned)"
                is_trained = False
                
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            return True, model_type, is_trained
        except Exception as e:
            return False, str(e), False

    def predict_text(self, text):
        if not self.model_loaded:
            return None
            
        processed_text = self.preprocess_text(text)
        
        if len(processed_text) < 10:
            return {'error': 'Text too short for analysis'}

        try:
            encoding = self.tokenizer(
                processed_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()

            return {
                'prediction': prediction,
                'prediction_label': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'real_prob': probabilities[0][0].item(),
                'fake_prob': probabilities[0][1].item()
            }
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}

def create_simple_chart(real_prob, fake_prob):
    """Simple chart that works reliably"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Real News', 'Fake News'],
        y=[real_prob, fake_prob],
        marker_color=['#10b981', '#ef4444'],
        text=[f"{real_prob:.3f}", f"{fake_prob:.3f}"],
        textposition='auto',
        textfont=dict(size=14, color='white')
    ))
    
    fig.update_layout(
        title="Analysis Results",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=40, l=40, r=40),
        font=dict(family="Inter")
    )
    
    fig.update_xaxes(title_text="Classification")
    fig.update_yaxes(title_text="Confidence Score", range=[0, 1])
    
    return fig

def main():
    # Initialize detector first
    if 'detector' not in st.session_state:
        st.session_state.detector = FakeNewsDetector()

    detector = st.session_state.detector

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        
        st.markdown("""
        **Fake News Detector** uses advanced AI to analyze news articles and detect potential misinformation.
        
        **Technology:**
        - BERT Transformer Model
        - Advanced NLP Processing
        - Real-time Analysis
        - Confidence Scoring
        """)
        
        st.markdown("---")
        
        if detector.model_loaded:
            st.markdown("**System Status**")
            selected_model = st.session_state.get('selected_model', 'Unknown')
            is_trained = st.session_state.get('is_trained', False)
            
            if is_trained:
                st.success(f"✅ {selected_model} Model Loaded")
                st.info("Fine-tuned for fake news detection")
            else:
                st.warning(f"⚠️ {selected_model} Loaded")
                st.info("Base BERT (not fine-tuned)")
                
            st.info(f"Device: {detector.device}")
            st.info(f"Max Tokens: {detector.max_length}")
        else:
            st.markdown("**System Status**")
            st.error("❌ No Model Loaded")
            st.info("Please select and load a model")
        
        st.markdown("---")
        
        st.markdown("**Best Practices**")
        st.markdown("""
        - Use complete articles (100+ words)
        - Always verify through multiple sources
        - Consider confidence levels
        - Apply critical thinking
        """)
        
        st.markdown("---")
        
        st.warning("""
        **Important Notice**
        
        This tool is for educational purposes. 
        Always verify news through reliable sources.
        AI predictions are not 100% accurate.
        """)

    # Header
    st.markdown("""
    <div class="header">
        <h1>Fake News Detector</h1>
        <p>AI-powered news authenticity verification</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Configuration
    st.markdown("### Model Setup")
    
    # Model selection dropdown
    model_options = {
        "RB-1 (Our Custom Trained Model)": "./fake_news_model",
        "Demo Model (BERT Base)": None
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            index=0,  # Default to F-1
            help="Choose between your trained model (F-1) or the demo BERT model"
        )
        
        model_path = model_options[selected_model]
        
        # Show the actual path for reference
        if model_path:
            st.caption(f"Model path: {model_path}")
        else:
            st.caption("Using pre-trained BERT (not fine-tuned for fake news)")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load Model"):
            with st.spinner(f"Loading {selected_model}..."):
                success, message, is_trained = detector.load_model(model_path)
                
                if success:
                    st.markdown(f'<div class="status-success">Model loaded: {selected_model}</div>', unsafe_allow_html=True)
                    st.session_state.model_type = f"{selected_model} ({message})"
                    st.session_state.is_trained = is_trained
                    st.session_state.selected_model = selected_model
                else:
                    st.markdown(f'<div class="status-error">Error loading {selected_model}: {message}</div>', unsafe_allow_html=True)

    # Model Status
    if detector.model_loaded:
        model_type = st.session_state.get('model_type', 'Unknown')
        is_trained = st.session_state.get('is_trained', False)
        
        if is_trained:
            st.markdown(f'<div class="status-success">Active: {model_type}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-warning">Active: {model_type} - For better accuracy, load a trained model</div>', unsafe_allow_html=True)

    # Input Section
    st.markdown("### Article Analysis")
    
    input_type = st.radio(
        "Choose input method:",
        ["Text Input", "Sample Articles"],
        horizontal=True
    )
    
    news_text = ""
    
    if input_type == "Text Input":
        news_text = st.text_area(
            "Enter news article:",
            height=150,
            placeholder="Paste your news article text here for analysis...",
            help="Enter the complete article text for best results"
        )
        
        if news_text:
            word_count = len(news_text.split())
            st.caption(f"{word_count} words • {len(news_text)} characters")
    
    else:
        sample_articles = {
            "Real News - Weather Update": "The National Weather Service has issued a comprehensive weather forecast predicting moderate to heavy rainfall across the southeastern United States this weekend. Meteorologists are advising residents to prepare for potential flooding in low-lying areas and to avoid unnecessary travel during peak storm hours.",
            
            "Fake News - Health Miracle": "Scientists at a secret underground research facility have allegedly discovered a revolutionary pill that allows people to lose 50 pounds in just one week without any changes to diet or exercise. According to unverified sources, this miraculous weight loss solution is being deliberately suppressed by major pharmaceutical companies.",
            
            "Real News - Tech Earnings": "Technology giant Apple Inc. reported stronger-than-expected quarterly earnings yesterday, with revenue increasing 15% year-over-year to reach $97.3 billion. The company attributed its strong performance to robust sales of the latest iPhone models and continued growth in its services division.",
            
            "Fake News - Government Conspiracy": "Leaked classified documents allegedly reveal that all smartphones manufactured since 2020 contain advanced microchips specifically designed to monitor and control human thoughts and behavior. According to these unverified claims, the technology is being used by a secret coalition of world governments."
        }
        
        selected_article = st.selectbox("Choose a sample article:", list(sample_articles.keys()))
        news_text = sample_articles[selected_article]
        
        st.text_area("Selected article:", value=news_text, height=100, disabled=True)

    # Analysis Button
    if st.button("Analyze Article", disabled=not detector.model_loaded):
        if not detector.model_loaded:
            st.error("Please load a model first!")
        elif not news_text.strip():
            st.warning("Please enter some text to analyze!")
        else:
            with st.spinner("Analyzing article..."):
                result = detector.predict_text(news_text)
            
            if result and 'error' not in result:
                # Main Prediction
                if result['prediction'] == 0:
                    st.markdown(f'''
                    <div class="prediction-real">
                        AUTHENTIC NEWS<br>
                        <small>Confidence: {result['confidence']:.1%}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-fake">
                        SUSPICIOUS NEWS<br>
                        <small>Confidence: {result['confidence']:.1%}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Metrics
                st.markdown("### Analysis Details")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-item">
                        <div class="metric-label">Real News</div>
                        <div class="metric-value" style="color: #10b981;">{result['real_prob']:.3f}</div>
                        <div class="progress-container">
                            <div class="progress-bar progress-real" style="width: {result['real_prob']*100}%;"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-item">
                        <div class="metric-label">Fake News</div>
                        <div class="metric-value" style="color: #ef4444;">{result['fake_prob']:.3f}</div>
                        <div class="progress-container">
                            <div class="progress-bar progress-fake" style="width: {result['fake_prob']*100}%;"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    confidence_color = "#10b981" if result['confidence'] > 0.7 else "#f59e0b" if result['confidence'] > 0.5 else "#ef4444"
                    
                    st.markdown(f'''
                    <div class="metric-item">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="color: {confidence_color};">{result['confidence']:.1%}</div>
                        <div class="progress-container">
                            <div class="progress-bar progress-confidence" style="width: {result['confidence']*100}%;"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Chart
                st.markdown("### Visualization")
                try:
                    fig = create_simple_chart(result['real_prob'], result['fake_prob'])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
                
                # Insights
                st.markdown("### AI Insights")
                
                if result['confidence'] < 0.6:
                    st.markdown('<div class="insight insight-info">Low confidence prediction. The AI is uncertain about this classification. Consider verifying through multiple reliable sources.</div>', unsafe_allow_html=True)
                elif result['prediction'] == 1 and result['confidence'] > 0.8:
                    st.markdown('<div class="insight insight-danger">High confidence fake news detected. Please verify through trusted sources before sharing.</div>', unsafe_allow_html=True)
                elif result['prediction'] == 0 and result['confidence'] > 0.8:
                    st.markdown('<div class="insight insight-success">High confidence authentic news. This appears to be legitimate content.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="insight insight-warning">Moderate confidence prediction. Consider cross-referencing with trusted sources.</div>', unsafe_allow_html=True)
            
            elif result and 'error' in result:
                st.error(f"Analysis Error: {result['error']}")
            else:
                st.error("Analysis failed. Please try again.")

    # Footer
    st.markdown(f'''
    <div class="footer">
        Developed by Dawood Ahmed • {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()