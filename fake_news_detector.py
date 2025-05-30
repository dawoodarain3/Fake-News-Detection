import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class NewsDataset(Dataset):
    """Custom Dataset for news articles"""

    def __init__(self, texts, labels, tokenizer, max_length=256):  
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FakeNewsDetector:
    """Main class for fake news detection"""

    def __init__(self, model_name='bert-base-uncased', max_length=256):  
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

        print(f"Using device: {self.device}")

    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print("Dataset file not found. Creating sample data for demonstration...")

            sample_data = {
                'text': [
                    # Fake news examples
                    "Scientists discover new cure for all diseases in breakthrough research that will end human suffering forever",
                    "Aliens land in major city, government covers up the incident and threatens witnesses with imprisonment",
                    "Miracle weight loss pill helps lose 50 pounds in one week without any side effects or diet changes",
                    "Secret society controls world governments from hidden underground location beneath major monuments",
                    "Doctors warn against drinking water as it causes instant death within minutes of consumption",
                    "Shocking discovery: The moon is actually made of cheese and NASA has been lying to us",
                    "New study proves that vegetables are actually harmful and cause cancer in 99% of people",
                    "Breaking: Time travel discovered by teenager in garage using household items and batteries",
                    "Billionaire secretly controls weather patterns to manipulate global economy for personal profit",
                    "Ancient pyramid contains technology that can solve world energy crisis but government hides it",
                    "Miracle herb cures diabetes, cancer, and heart disease overnight according to this one weird trick",
                    "Scientists confirm that Earth is actually flat and all photos from space are computer generated",
                    "Local man discovers fountain of youth in backyard, age reversal confirmed by independent studies",
                    "Government admits to mind control experiments through television broadcasts and radio waves",
                    "Shocking revelation: All major celebrities are actually robots controlled by shadow organizations",
                    "New research shows breathing air is dangerous and we should hold our breath for better health",
                    "Breaking news: Gravity doesn't exist and objects fall due to invisible alien manipulation",
                    "Miracle supplement allows people to survive without food or water for months at a time",
                    "Scientists discover parallel universe where everything is opposite and people live backwards",
                    "Government secretly replaced all birds with surveillance drones in massive conspiracy operation",

                    # Real news examples
                    "Local weather forecast shows sunny skies for the weekend with temperatures reaching 75 degrees",
                    "Stock market opens higher after positive economic indicators from quarterly earnings reports",
                    "New study reveals benefits of regular exercise for heart health and cardiovascular improvement",
                    "Technology company announces new smartphone with improved battery life and camera features",
                    "University researchers publish findings on climate change effects in peer-reviewed journal",
                    "City council approves new public transportation initiative to reduce traffic congestion downtown",
                    "Local hospital receives federal funding for new medical equipment and facility improvements",
                    "Education department announces increased funding for public schools and teacher training programs",
                    "Agricultural researchers develop drought-resistant crops to help farmers in arid regions",
                    "Sports team wins championship game with final score of 28-21 in overtime victory",
                    "Library system expands digital collection and adds new community programming for all ages",
                    "Road construction project begins next month to improve highway safety and reduce accidents",
                    "University offers new scholarship program for first-generation college students from local area",
                    "Recycling program expansion allows residents to dispose of electronic waste at collection centers",
                    "Police department implements new community outreach program to improve neighborhood relations",
                    "Local business owners report increased sales following recent downtown revitalization project",
                    "Museum opens new exhibit featuring artifacts from local historical society collection",
                    "Park service announces trail maintenance schedule and temporary closures for visitor safety",
                    "School district implements new after-school tutoring program to help students improve grades",
                    "Fire department responds to house fire on Main Street, no injuries reported in incident"
                ],
                'label': [1]*20 + [0]*20  # 1 = Fake, 0 = Real
            }
            df = pd.DataFrame(sample_data)

        df['text'] = df['text'].astype(str)
        df = df.dropna()

        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        return df

    def create_data_loaders(self, df, batch_size=8, test_size=0.3):  
        """Create train and validation data loaders"""

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=test_size,
            random_state=42,
            stratify=df['label'].values
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")

        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, train_loader, val_loader, epochs=20
                    , learning_rate=2e-5):  
        """Train the BERT model"""

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)  
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  
            num_training_steps=total_steps
        )

        train_losses = []
        val_accuracies = []
        val_losses = []
        best_accuracy = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training phase
            self.model.train()
            total_train_loss = 0

            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            val_accuracy, val_loss = self.evaluate_model(val_loader)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best accuracy: {best_accuracy:.4f}")

        return train_losses, val_losses, val_accuracies

    def evaluate_model(self, val_loader):
        """Evaluate the model on validation set"""
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        avg_loss = total_loss / len(val_loader)
        return accuracy, avg_loss

    def predict_single_text(self, text):
        """Predict if a single text is fake or real"""
        self.model.eval()

        encoding = self.tokenizer(
            text,
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
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'probabilities': {
                'Real': probabilities[0][0].item(),
                'Fake': probabilities[0][1].item()
            }
        }

    def save_model(self, path):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model"""
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")

def plot_training_results(train_losses, val_losses, val_accuracies):
    """Plot training results with separate subplots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, 'b-o', linewidth=2, markersize=6)
    axes[0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')
    
    # Plot 2: Validation Loss
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, 'r-s', linewidth=2, markersize=6)
    axes[1].set_title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')
    
    # Plot 3: Validation Accuracy
    axes[2].plot(range(1, len(val_accuracies) + 1), val_accuracies, 'g-^', linewidth=2, markersize=6)
    axes[2].set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_detailed(detector, val_loader):
    """Detailed evaluation with confusion matrix and classification report"""
    detector.model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(detector.device)
            attention_mask = batch['attention_mask'].to(detector.device)
            labels = batch['labels'].to(detector.device)

            outputs = detector.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    print("\n" + "="*50)
    print("DETAILED EVALUATION RESULTS")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions,
                              target_names=['Real', 'Fake'],
                              digits=4))

    # Confusion Matrix with better styling
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8})
    
    plt.title('Confusion Matrix\nFake News Detection Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy = accuracy_score(true_labels, predictions)
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the fake news detection pipeline"""
    print("="*60)
    print("FAKE NEWS DETECTION TRAINING PIPELINE")
    print("="*60)

    detector = FakeNewsDetector()

    df = detector.load_and_preprocess_data('fake_news_dataset.csv')

    train_loader, val_loader = detector.create_data_loaders(df, batch_size=4)  

    print("\nStarting training...")
    train_losses, val_losses, val_accuracies = detector.train_model(
        train_loader, val_loader, epochs=20, learning_rate=2e-5
    )

    # Plot results
    plot_training_results(train_losses, val_losses, val_accuracies)

    print("\nDetailed Evaluation:")
    evaluate_detailed(detector, val_loader)

    # Test on sample texts
    test_texts = [
        "Breaking: Scientists discover that drinking water can be harmful to your health and causes instant death",
        "The weather forecast predicts rain for tomorrow in the city with temperatures around 65 degrees",
        "Miracle pill allows people to lose 30 pounds in 3 days without exercise or diet changes",
        "Local university announces new scholarship program for students in the community"
    ]

    print("\n" + "="*70)
    print("TESTING ON SAMPLE TEXTS")
    print("="*70)

    for i, text in enumerate(test_texts, 1):
        result = detector.predict_single_text(text)
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        print(f"Probabilities: Real={result['probabilities']['Real']:.3f}, "
              f"Fake={result['probabilities']['Fake']:.3f}")
        print("-" * 70)

    # Save model
    detector.save_model('./fake_news_model')
    print("\nTraining completed successfully!")
    print("Model saved to './fake_news_model'")

if __name__ == "__main__":
    main()