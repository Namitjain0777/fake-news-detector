import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Enhanced NLTK data download with better error handling
def setup_nltk():
    """Setup NLTK with proper error handling and fallbacks"""
    required_datasets = [
        'punkt',
        'punkt_tab',  # New tokenizer for newer NLTK versions
        'stopwords',
        'vader_lexicon'
    ]
    
    for dataset in required_datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"✅ Downloaded/Verified: {dataset}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {dataset}: {str(e)}")
    
    print("📦 NLTK setup completed!")

# Alternative tokenizer function that doesn't rely on NLTK punkt
def simple_tokenize(text):
    """Simple tokenizer that doesn't require NLTK punkt"""
    if pd.isna(text):
        return []
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    
    # Split by whitespace and filter short words
    tokens = [word.strip() for word in text.split() if len(word.strip()) > 2]
    
    return tokens

class FakeNewsDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.stop_words = None
        self.stemmer = PorterStemmer()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize stop words with fallback
        self._setup_stopwords()
        
    def _setup_stopwords(self):
        """Setup stopwords with fallback options"""
        try:
            self.stop_words = set(stopwords.words('english'))
            print("✅ Using NLTK stopwords")
        except Exception as e:
            print(f"⚠️  NLTK stopwords not available: {e}")
            # Fallback to basic English stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                'with', 'through', 'during', 'before', 'after', 'above', 'below',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
            print("✅ Using fallback stopwords list")
        
    def load_data(self, true_path, fake_path):
        """Load and combine true and fake news datasets"""
        print("📊 STEP 1: LOADING DATA")
        print("=" * 50)
        
        try:
            # Load datasets
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            
            # Add labels
            true_df['label'] = 1  # Real news
            fake_df['label'] = 0  # Fake news
            
            # Combine datasets
            self.data = pd.concat([true_df, fake_df], ignore_index=True)
            
            print(f"✅ True news articles loaded: {len(true_df)}")
            print(f"✅ Fake news articles loaded: {len(fake_df)}")
            print(f"✅ Total articles: {len(self.data)}")
            print(f"✅ Dataset columns: {list(self.data.columns)}")
            
            return self.data
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {str(e)}")
            print("💡 Tip: Make sure your file paths are correct:")
            print(f"   True path: {true_path}")
            print(f"   Fake path: {fake_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def explore_data(self):
        """Comprehensive data exploration and analysis"""
        print("\n📈 STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        # Basic information
        print("📋 Dataset Information:")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage().sum() / 1024**2:.2f} MB")
        print("\nData types:")
        print(self.data.dtypes)
        
        # Missing values
        print("\n🔍 Missing Values Analysis:")
        missing_data = self.data.isnull().sum()
        missing_found = missing_data[missing_data > 0]
        if len(missing_found) > 0:
            print(missing_found)
        else:
            print("✅ No missing values found!")
        
        # Label distribution
        print("\n📊 Label Distribution:")
        label_counts = self.data['label'].value_counts()
        print(f"Real news (1): {label_counts[1]} ({label_counts[1]/len(self.data)*100:.1f}%)")
        print(f"Fake news (0): {label_counts[0]} ({label_counts[0]/len(self.data)*100:.1f}%)")
        
        # Visualizations
        try:
            self._create_visualizations()
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            print("Continuing without visualizations...")
        
    def _create_visualizations(self):
        """Create visualizations for data analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Label distribution
        self.data['label'].value_counts().plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Distribution of Real vs Fake News')
        axes[0,0].set_xlabel('Label (0: Fake, 1: Real)')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_xticklabels(['Fake', 'Real'], rotation=0)
        
        # 2. Text length analysis
        text_col = self._get_text_column()
        if text_col:
            self.data['text_length'] = self.data[text_col].astype(str).str.len()
            self.data.boxplot(column='text_length', by='label', ax=axes[0,1])
            axes[0,1].set_title('Text Length Distribution by Label')
            axes[0,1].set_xlabel('Label (0: Fake, 1: Real)')
            axes[0,1].set_ylabel('Text Length')
        
        # 3. Word count analysis
        if text_col:
            self.data['word_count'] = self.data[text_col].astype(str).str.split().str.len()
            self.data.boxplot(column='word_count', by='label', ax=axes[1,0])
            axes[1,0].set_title('Word Count Distribution by Label')
            axes[1,0].set_xlabel('Label (0: Fake, 1: Real)')
            axes[1,0].set_ylabel('Word Count')
        
        # 4. Missing values heatmap
        sns.heatmap(self.data.isnull(), ax=axes[1,1], cbar=True, yticklabels=False)
        axes[1,1].set_title('Missing Values Heatmap')
        
        plt.tight_layout()
        plt.show()
        
        # Word clouds
        try:
            self._create_wordclouds()
        except Exception as e:
            print(f"⚠️  Word cloud creation failed: {e}")
    
    def _get_text_column(self):
        """Identify the main text column"""
        possible_cols = ['text', 'content', 'article', 'body', 'news', 'title']
        for col in possible_cols:
            if col in self.data.columns:
                return col
        
        # If no standard column found, use the first string column
        string_cols = self.data.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            # Exclude the label column
            text_cols = [col for col in string_cols if col != 'label']
            return text_cols[0] if text_cols else None
        
        return None
    
    def _create_wordclouds(self):
        """Create word clouds for real and fake news"""
        text_col = self._get_text_column()
        if not text_col:
            print("❌ No text column found for word cloud generation")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        try:
            # Real news word cloud
            real_text = ' '.join(self.data[self.data['label'] == 1][text_col].dropna().astype(str))
            if real_text.strip():
                real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_text)
                axes[0].imshow(real_wordcloud, interpolation='bilinear')
                axes[0].set_title('Real News Word Cloud', fontsize=16)
                axes[0].axis('off')
            
            # Fake news word cloud
            fake_text = ' '.join(self.data[self.data['label'] == 0][text_col].dropna().astype(str))
            if fake_text.strip():
                fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
                axes[1].imshow(fake_wordcloud, interpolation='bilinear')
                axes[1].set_title('Fake News Word Cloud', fontsize=16)
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Word cloud generation failed: {e}")
    
    def preprocess_text(self, text):
        """Comprehensive text preprocessing with improved error handling"""
        if pd.isna(text) or text is None:
            return ""
        
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Try NLTK tokenization first, fall back to simple tokenization
            try:
                tokens = word_tokenize(text)
            except Exception:
                # Fallback to simple tokenization
                tokens = simple_tokenize(text)
            
            # Remove stopwords and stem
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        # Try stemming
                        stemmed = self.stemmer.stem(token)
                        processed_tokens.append(stemmed)
                    except Exception:
                        # If stemming fails, use original token
                        processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"⚠️  Error processing text: {e}")
            # Return cleaned text without advanced processing
            return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    
    def prepare_data(self):
        """Prepare data for machine learning"""
        print("\n🔧 STEP 3: DATA PREPROCESSING")
        print("=" * 50)
        
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        text_col = self._get_text_column()
        if not text_col:
            print("❌ No text column found for preprocessing")
            return
        
        print(f"📝 Processing text from column: {text_col}")
        
        # Handle missing values
        self.data[text_col] = self.data[text_col].fillna('').astype(str)
        
        # Preprocess text
        print("🔄 Applying text preprocessing...")
        try:
            self.data['processed_text'] = self.data[text_col].apply(self.preprocess_text)
            print("✅ Text preprocessing completed successfully!")
        except Exception as e:
            print(f"❌ Error during text preprocessing: {e}")
            return
        
        # Remove empty texts
        original_size = len(self.data)
        self.data = self.data[self.data['processed_text'] != '']
        removed_count = original_size - len(self.data)
        
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} empty texts")
        
        print(f"✅ Preprocessing complete. Final dataset size: {len(self.data)}")
        
        # Feature extraction
        print("🔢 Extracting features using TF-IDF...")
        try:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(self.data['processed_text'])
            y = self.data['label']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"✅ Train set size: {self.X_train.shape}")
            print(f"✅ Test set size: {self.X_test.shape}")
            print(f"✅ Features extracted: {self.X_train.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            return
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n🤖 STEP 4: MODEL TRAINING")
        print("=" * 50)
        
        if self.X_train is None:
            print("❌ No preprocessed data available. Please prepare data first.")
            return
        
        # Define models
        models_config = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        print("🔄 Training models...")
        
        for name, model in models_config.items():
            try:
                print(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"✅ {name} trained successfully!")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        if self.models:
            print(f"✅ {len(self.models)} models trained successfully!")
        else:
            print("❌ No models were trained successfully.")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📊 STEP 5: MODEL EVALUATION")
        print("=" * 50)
        
        if not self.models:
            print("❌ No trained models available. Please train models first.")
            return
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                results[name] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"\n🎯 {name} Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"AUC Score: {auc_score:.4f}")
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred, 
                                          target_names=['Fake', 'Real']))
            except Exception as e:
                print(f"❌ Error evaluating {name}: {e}")
        
        # Visualization
        if results:
            try:
                self._plot_model_comparison(results)
                self._plot_confusion_matrices(results)
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")
        
        return results
    
    def _plot_model_comparison(self, results):
        """Plot model comparison"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        aucs = [results[model]['auc'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        colors = ['blue', 'green', 'orange', 'red'][:len(models)]
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC comparison
        bars2 = ax2.bar(models, aucs, color=colors, alpha=0.7)
        ax2.set_title('Model AUC Score Comparison')
        ax2.set_ylabel('AUC Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        num_models = len(results)
        cols = min(2, num_models)
        rows = (num_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
        if num_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if num_models > 1 else [axes]
        else:
            axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], 
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        # Hide extra subplots if any
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def predict_news(self, text, model_name='Logistic Regression'):
        """Predict if a news article is fake or real"""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not available. Available models: {list(self.models.keys())}")
            return None
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                print("⚠️  Warning: Processed text is empty")
                return None
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict
            model = self.models[model_name]
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector)[0]
            
            result = {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': max(probability) * 100,
                'probabilities': {
                    'fake': probability[0] * 100,
                    'real': probability[1] * 100
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def analyze_feature_importance(self, model_name='Logistic Regression'):
        """Analyze feature importance"""
        print("\n🔍 STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not available.")
            return
        
        try:
            model = self.models[model_name]
            feature_names = self.vectorizer.get_feature_names_out()
            
            if hasattr(model, 'coef_'):
                # For linear models
                coefficients = model.coef_[0]
                
                # Get top features for fake news (negative coefficients)
                fake_indices = np.argsort(coefficients)[:20]
                fake_features = [(feature_names[i], coefficients[i]) for i in fake_indices]
                
                # Get top features for real news (positive coefficients)
                real_indices = np.argsort(coefficients)[-20:]
                real_features = [(feature_names[i], coefficients[i]) for i in real_indices]
                
                print("🔴 Top words indicating FAKE news:")
                for feature, coef in fake_features:
                    print(f"  {feature}: {coef:.4f}")
                
                print("\n🟢 Top words indicating REAL news:")
                for feature, coef in reversed(real_features):
                    print(f"  {feature}: {coef:.4f}")
                    
            elif hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                indices = np.argsort(importances)[-20:]
                
                print("🔍 Top 20 most important features:")
                for i in reversed(indices):
                    print(f"  {feature_names[i]}: {importances[i]:.4f}")
                    
        except Exception as e:
            print(f"❌ Error analyzing feature importance: {e}")

# Example usage and demonstration
def main():
    """Main function to demonstrate the fake news detector"""
    
    print("🚀 FAKE NEWS DETECTOR WITH COMPREHENSIVE DATA ANALYSIS")
    print("=" * 60)
    
    # Setup NLTK first
    setup_nltk()
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # File paths - Update these to match your actual file paths
    true_path = "/workspaces/Fakee-news-detector/data sets/fake.csv.csv"  # Update this path
    fake_path = "/workspaces/Fakee-news-detector/data sets/true.csv.csv"  # Update this path
    
    print(f"\n📁 Looking for data files:")
    print(f"   True news: {true_path}")
    print(f"   Fake news: {fake_path}")
    
    try:
        # Step 1: Load data
        data = detector.load_data(true_path, fake_path)
        if data is None:
            print("\n💡 Quick Setup Guide:")
            print("1. Download the dataset from Kaggle: 'Fake and real news dataset'")
            print("2. Place 'True.csv' and 'Fake.csv' in the same directory as this script")
            print("3. Or update the file paths in the code above")
            return
        
        # Step 2: Explore data
        detector.explore_data()
        
        # Step 3: Prepare data
        detector.prepare_data()
        
        # Step 4: Train models
        detector.train_models()
        
        # Step 5: Evaluate models
        results = detector.evaluate_models()
        
        # Step 6: Feature importance
        if detector.models:
            detector.analyze_feature_importance()
        
        # Example prediction
        sample_text = """
        Scientists at MIT have developed a new artificial intelligence system 
        that can detect fake news with 95% accuracy. The system uses advanced 
        natural language processing techniques to analyze text patterns and 
        cross-reference information with reliable sources.
        """
        
        print("\n🔮 SAMPLE PREDICTION")
        print("=" * 50)
        result = detector.predict_news(sample_text)
        if result:
            print(f"Text: {sample_text[:100]}...")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Probabilities: Fake: {result['probabilities']['fake']:.1f}%, Real: {result['probabilities']['real']:.1f}%")
            
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\n💡 Setup Instructions:")
        print("1. Download the Fake News dataset from Kaggle")
        print("2. Extract the CSV files (True.csv and Fake.csv)")
        print("3. Update the file paths in the code or place files in the same directory")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the error details above and try again.")

if __name__ == "__main__":
    main()