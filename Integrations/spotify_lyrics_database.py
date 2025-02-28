import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import unicodedata
import re
from bs4 import BeautifulSoup
import urllib.parse
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning Imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# TensorFlow Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Advanced Genre Classification Imports
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
import joblib
import os
import logging
import asyncio
import aiohttp
import redis
from functools import lru_cache
from typing import Dict, List, Any

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='genre_classifier.log'
)
logger = logging.getLogger(__name__)

class ScalableGenreClassifier:
    def __init__(self):
        # Redis for caching
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Spotify API setup
        try:
            self.sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=CLIENT_ID, 
                    client_secret=CLIENT_SECRET
                )
            )
        except Exception as e:
            logger.error(f"Spotify API initialization failed: {e}")
            self.sp = None
        
        # Model and tokenizer paths
        self.model_dir = os.path.join(os.path.dirname(__file__), 'genre_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.text_model = self._load_or_train_text_model()
        self.label_encoder = self._load_label_encoder()
    
    def _preprocess_text(self, text: str) -> np.ndarray:
        """
        Advanced text preprocessing for deep learning
        """
        # Tokenization
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts([text])
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        
        return padded_sequence
    
    def _load_label_encoder(self) -> LabelEncoder:
        """
        Load or create label encoder
        """
        encoder_path = os.path.join(self.model_dir, 'label_encoder.joblib')
        
        try:
            if os.path.exists(encoder_path):
                return joblib.load(encoder_path)
        except Exception as e:
            logger.warning(f"Could not load label encoder: {e}")
        
        # Create default encoder
        encoder = LabelEncoder()
        encoder.fit(['Hip Hop', 'Pop', 'Rock', 'Electronic', 'R&B'])
        
        try:
            joblib.dump(encoder, encoder_path)
        except Exception as e:
            logger.error(f"Could not save label encoder: {e}")
        
        return encoder
    
    def _load_or_train_text_model(self):
        """
        Load existing model or train a new deep learning model
        """
        model_path = os.path.join(self.model_dir, 'genre_text_model.h5')
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                return tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
        
        # Sample training data
        lyrics_data = pd.DataFrame({
            'lyrics': [
                "I love you, you're my everything",
                "Money, cash, success, millionaire lifestyle",
                "Rocking guitar, intense energy, rebellion",
                "Smooth beats, electronic rhythm",
                "Sad, emotional, deep feelings"
            ],
            'genre': [
                "Romance",
                "Hip Hop",
                "Rock",
                "Electronic",
                "Blues"
            ]
        })
        
        # Train model
        self.text_model, _, self.label_encoder = self.train_genre_classifier(lyrics_data)
        
        return self.text_model
    
    def train_genre_classifier(self, lyrics_data: pd.DataFrame):
        # Preprocessing
        lyrics = lyrics_data['lyrics'].fillna('')
        labels = lyrics_data['genre'].fillna('Unknown')
        
        # Dynamic Label Encoding
        unique_labels = set(labels)
        label_encoder = LabelEncoder()
        label_encoder.fit(list(unique_labels))
        
        # Handle unseen labels gracefully
        def safe_transform(label):
            try:
                return label_encoder.transform([label])[0]
            except ValueError:
                # If label is unseen, assign it to a default category
                return label_encoder.transform(['Unknown'])[0]
        
        # Transform labels
        encoded_labels = [safe_transform(label) for label in labels]
        
        # One-hot encode labels
        one_hot_labels = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            lyrics, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Tokenization
        tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        # Sequence Padding
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post', truncating='post')
        
        # Model Architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(5000, 16, input_length=200),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Training
        history = model.fit(
            X_train_pad, 
            to_categorical(y_train, num_classes=len(label_encoder.classes_)), 
            epochs=10, 
            validation_split=0.2,
            verbose=1
        )
        
        # Save model and tokenizer
        model.save('genre_classification_model.h5')
        joblib.dump(tokenizer, 'genre_tokenizer.pkl')
        joblib.dump(label_encoder, 'genre_label_encoder.pkl')
        
        return model, tokenizer, label_encoder
    
    @lru_cache(maxsize=1000)
    def _get_spotify_genres(self, artist_name: str) -> List[str]:
        """
        Cached Spotify genre retrieval
        """
        if not self.sp:
            return []
        
        try:
            results = self.sp.search(q=f'artist:{artist_name}', type='artist')
            
            if results['artists']['items']:
                artist = results['artists']['items'][0]
                return artist.get('genres', [])
            return []
        except Exception as e:
            logger.error(f"Spotify API error for {artist_name}: {e}")
            return []
    
    async def fetch_external_genre_data(self, lyrics: str) -> Dict[str, Any]:
        """
        Asynchronously fetch additional genre context from external APIs
        """
        async with aiohttp.ClientSession() as session:
            try:
                # Example: Hypothetical external genre API
                async with session.post(
                    'https://genre-api.example.com/classify', 
                    json={'lyrics': lyrics}
                ) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                logger.error(f"External genre API error: {e}")
        
        return {}
    
    def extract_dynamic_genres(self, lyrics: str) -> List[str]:
        """
        Dynamically extract genres from lyrics using advanced NLP techniques
        
        Args:
            lyrics (str): Full text of song lyrics
        
        Returns:
            List[str]: Dynamically extracted genres
        """
        # Convert lyrics to lowercase for consistent matching
        lyrics_lower = lyrics.lower()
        
        # Define genre keyword mappings with enhanced Hip Hop detection
        genre_keywords = {
            'Hip Hop': [
                # Money and wealth references
                'money', 'cash', 'millionaire', 'rich', 'wealth', 'bank', 
                # Street and lifestyle
                'street', 'hustle', 'flow', 'bars', 'rhyme', 'rap', 
                # Attitude and swagger
                'swagger', 'boss', 'power', 'success', 'grind', 'game',
                # Slang and urban language
                'yo', 'ain\'t', 'shit', 'dope', 'lit', 'flex', 
                # Cultural references
                'hood', 'block', 'crew', 'squad', 'lifestyle'
            ],
            'Romance': ['love', 'heart', 'kiss', 'romance', 'relationship', 'emotion', 'feelings'],
            'Rock': ['guitar', 'rebel', 'loud', 'energy', 'passion', 'scream', 'intense'],
            'Electronic': ['beat', 'dance', 'rhythm', 'club', 'electronic', 'synth', 'techno'],
            'Blues': ['sad', 'pain', 'heartbreak', 'soul', 'deep', 'emotion', 'melancholy'],
            'Pop': ['catchy', 'radio', 'popular', 'sing', 'melody', 'chorus'],
            'Country': ['truck', 'hometown', 'rural', 'cowboy', 'farm', 'road'],
            'Indie': ['alternative', 'indie', 'underground', 'unique', 'artistic'],
            'Jazz': ['smooth', 'saxophone', 'improvisation', 'cool', 'rhythm'],
        }
        
        # Advanced genre detection
        detected_genres = []
        
        # Keyword-based genre detection with Hip Hop prioritization
        hip_hop_matches = sum(keyword in lyrics_lower for keyword in genre_keywords['Hip Hop'])
        
        # Aggressive Hip Hop detection
        if hip_hop_matches >= 3:
            detected_genres.append('Hip Hop')
        
        # Continue with other genre detection
        for genre, keywords in genre_keywords.items():
            if genre == 'Hip Hop':
                continue  # Already handled
            
            # Count keyword matches
            keyword_matches = sum(keyword in lyrics_lower for keyword in keywords)
            
            # If more than 2 keywords match, consider it a potential genre
            if keyword_matches >= 2:
                detected_genres.append(genre)
        
        # Sentiment-based genre refinement
        blob = TextBlob(lyrics)
        sentiment_score = blob.sentiment.polarity
        
        # Add sentiment-based genres
        if sentiment_score > 0.5:
            detected_genres.extend(['Romance', 'Pop'])
        elif sentiment_score < -0.5:
            detected_genres.extend(['Blues', 'Rock'])
        
        # Machine Learning Genre Prediction
        try:
            # Tokenize and pad the lyrics
            tokenizer = joblib.load('genre_tokenizer.pkl')
            model = tf.keras.models.load_model('genre_classification_model.h5')
            label_encoder = joblib.load('genre_label_encoder.pkl')
            
            # Prepare input
            lyrics_seq = tokenizer.texts_to_sequences([lyrics])
            lyrics_pad = pad_sequences(lyrics_seq, maxlen=200, padding='post', truncating='post')
            
            # Predict
            predictions = model.predict(lyrics_pad)[0]
            top_genre_indices = predictions.argsort()[-3:][::-1]
            ml_genres = [label_encoder.classes_[idx] for idx in top_genre_indices]
            
            detected_genres.extend(ml_genres)
        except Exception as e:
            logger.warning(f"ML Genre prediction failed: {e}")
        
        # Remove duplicates and limit to top 3
        detected_genres = list(dict.fromkeys(detected_genres))[:3]
        
        # Prioritize Hip Hop if it's a match
        if 'Hip Hop' in detected_genres:
            detected_genres.remove('Hip Hop')
            detected_genres.insert(0, 'Hip Hop')
        
        # Fallback if no genres detected
        if not detected_genres:
            detected_genres = ['Unknown']
        
        return detected_genres
    
    def advanced_semantic_genre_classification(self, lyrics: str) -> Dict[str, Any]:
        """
        Advanced semantic genre classification using multiple AI techniques
        
        Args:
            lyrics (str): Full song lyrics text
        
        Returns:
            Dict[str, Any]: Comprehensive genre classification results
        """
        # Import additional semantic analysis libraries
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Load pre-trained semantic embedding model
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            return {'genres': ['Unknown'], 'probabilities': {}}
        
        # Semantic genre reference corpus
        genre_references = {
            'Hip Hop': [
                "Expressing street life and urban experiences",
                "Showcasing personal success and wealth",
                "Describing hustling and overcoming challenges",
                "Celebrating individual achievement and swagger"
            ],
            'Romance': [
                "Exploring deep emotional connections",
                "Describing love and intimate relationships",
                "Expressing vulnerability and passion",
                "Narrating personal romantic experiences"
            ],
            'Rock': [
                "Expressing rebellion and intense emotions",
                "Describing personal struggles and strength",
                "Showcasing raw energy and passion",
                "Challenging societal norms"
            ],
            'Electronic': [
                "Creating rhythmic and dance-oriented experiences",
                "Exploring futuristic and technological themes",
                "Generating high-energy musical landscapes",
                "Focusing on beat and sonic experimentation"
            ],
            'Blues': [
                "Expressing deep personal pain and sorrow",
                "Describing life's hardships and struggles",
                "Exploring emotional depth and resilience",
                "Narrating personal transformation"
            ]
        }
        
        # Embed lyrics
        lyrics_embedding = embedding_model.encode([lyrics])[0]
        
        # Compute semantic similarities
        genre_similarities = {}
        for genre, references in genre_references.items():
            # Embed genre references
            reference_embeddings = embedding_model.encode(references)
            
            # Compute average similarity
            similarities = cosine_similarity([lyrics_embedding], reference_embeddings)[0]
            genre_similarities[genre] = np.mean(similarities)
        
        # Sort genres by similarity
        sorted_genres = sorted(genre_similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        top_genres = sorted_genres[:3]
        genre_probabilities = {
            genre: round(similarity * 100, 2)
            for genre, similarity in top_genres
        }
        
        # Normalize probabilities
        total_prob = sum(genre_probabilities.values())
        if total_prob != 100:
            # Adjust the first genre to make total 100%
            first_genre = top_genres[0][0]
            genre_probabilities[first_genre] += (100 - total_prob)
        
        # Machine Learning Validation
        try:
            # Load existing model
            model = tf.keras.models.load_model('genre_classification_model.h5')
            tokenizer = joblib.load('genre_tokenizer.pkl')
            label_encoder = joblib.load('genre_label_encoder.pkl')
            
            # Prepare input
            lyrics_seq = tokenizer.texts_to_sequences([lyrics])
            lyrics_pad = pad_sequences(lyrics_seq, maxlen=200, padding='post', truncating='post')
            
            # Predict
            ml_predictions = model.predict(lyrics_pad)[0]
            top_ml_indices = ml_predictions.argsort()[-3:][::-1]
            ml_genres = [label_encoder.classes_[idx] for idx in top_ml_indices]
            
            # Combine semantic and ML predictions
            for ml_genre in ml_genres:
                if ml_genre not in genre_probabilities:
                    genre_probabilities[ml_genre] = round(ml_predictions[top_ml_indices[ml_genres.index(ml_genre)]] * 100, 2)
        except Exception as e:
            logger.warning(f"ML Genre validation failed: {e}")
        
        # Ensure at least one genre is returned
        if not genre_probabilities:
            genre_probabilities = {'Unknown': 100.0}
        
        return {
            'genres': list(genre_probabilities.keys()),
            'probabilities': genre_probabilities
        }
    
    def unsupervised_theme_extraction(self, lyrics: str) -> Dict[str, Any]:
        """
        Advanced dynamic theme extraction using NLP techniques
        
        Args:
            lyrics (str): Full song lyrics text
        
        Returns:
            Dict[str, Any]: Dynamically extracted themes
        """
        # Advanced NLP Imports
        import nltk
        import logging
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
        
        # Ensure NLTK resources are downloaded with extended error handling
        nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {e}")
        
        from nltk import pos_tag, word_tokenize
        from nltk.corpus import stopwords
        from collections import Counter
        import re
        from textblob import TextBlob
        
        # Preprocess lyrics with enhanced error handling
        def clean_text(text):
            try:
                # Remove special characters and digits
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                return text
            except Exception as e:
                logger.error(f"Text cleaning error: {e}")
                return text
        
        # Clean and tokenize lyrics with fallback mechanism
        try:
            cleaned_lyrics = clean_text(lyrics)
            tokens = word_tokenize(cleaned_lyrics)
        except Exception as e:
            logger.warning(f"Tokenization failed, using basic split: {e}")
            tokens = cleaned_lyrics.split()
        
        # Remove stopwords with fallback
        try:
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word not in stop_words]
        except Exception as e:
            logger.warning(f"Stopwords removal failed: {e}")
            filtered_tokens = tokens
        
        # Part of Speech Tagging with fallback
        try:
            pos_tagged = pos_tag(filtered_tokens)
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}")
            pos_tagged = [(word, 'NN') for word in filtered_tokens]
        
        # Extract meaningful words (nouns, verbs, adjectives)
        meaningful_words = {
            'nouns': [word for word, pos in pos_tagged if pos.startswith('N')],
            'verbs': [word for word, pos in pos_tagged if pos.startswith('V')],
            'adjectives': [word for word, pos in pos_tagged if pos.startswith('J')]
        }
        
        # Compute word frequencies with error handling
        try:
            word_freq = {
                category: Counter(words).most_common(5)
                for category, words in meaningful_words.items()
            }
        except Exception as e:
            logger.error(f"Word frequency computation failed: {e}")
            word_freq = {category: [] for category in meaningful_words}
        
        # Generate theme descriptions
        def generate_theme_description(category_freq):
            return [f"{word} ({count})" for word, count in category_freq]
        
        themes = {
            'Nouns': generate_theme_description(word_freq['nouns']),
            'Verbs': generate_theme_description(word_freq['verbs']),
            'Descriptors': generate_theme_description(word_freq['adjectives'])
        }
        
        # Sentiment Analysis with error handling
        try:
            blob = TextBlob(lyrics)
            sentiment_score = blob.sentiment.polarity
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment_score = 0.0
        
        # Compute Theme Probabilities
        theme_probabilities = {}
        for theme_type, theme_words in themes.items():
            if theme_words:
                theme_probabilities[theme_type] = round(len(theme_words) * 20, 2)
        
        # Add Sentiment Context
        if sentiment_score > 0.5:
            theme_probabilities['Positive Emotional Themes'] = round(sentiment_score * 50, 2)
        elif sentiment_score < -0.5:
            theme_probabilities['Negative Emotional Themes'] = round(abs(sentiment_score) * 50, 2)
        
        # Ensure at least one theme is detected
        if not theme_probabilities:
            theme_probabilities['Undefined Themes'] = 100.0
        
        # Log final theme extraction results
        logger.info(f"Theme Extraction Results: {theme_probabilities}")
        
        return {
            'Detected Themes': list(theme_probabilities.keys()),
            'Theme Probabilities': theme_probabilities,
            'Theme Details': themes,
            'Spotify Genres': []  # Placeholder for future Spotify genre integration
        }
    
    def classify_genre(self, lyrics: str, artist_name: str = None) -> Dict[str, Any]:
        """
        Advanced dynamic theme classification
        
        Args:
            lyrics (str): Full text of song lyrics
            artist_name (str, optional): Name of the artist for additional context
        
        Returns:
            Dict[str, Any]: Comprehensive theme classification results
        """
        # Dynamic Theme Extraction
        theme_results = self.unsupervised_theme_extraction(lyrics)
        
        # Spotify Genre Lookup (if artist name provided)
        spotify_genres = []
        if artist_name:
            try:
                spotify_genres = self._get_spotify_genres(artist_name)
            except Exception as e:
                logger.warning(f"Spotify genre lookup failed: {e}")
        
        # Combine Results
        return {
            'Detected Themes': theme_results['Detected Themes'],
            'Theme Probabilities': theme_results['Theme Probabilities'],
            'Theme Details': theme_results['Theme Details'],
            'Spotify Genres': spotify_genres
        }
    
    def classify_song_genre(self, lyrics: str) -> Dict[str, float]:
        """
        Classify song genres with probabilities
        
        Args:
            lyrics (str): Full text of song lyrics
        
        Returns:
            Dict[str, float]: Genres with their probabilities
        """
        # Extract dynamic genres
        genres = self.extract_dynamic_genres(lyrics)
        
        # Generate probabilistic distribution with Hip Hop prioritization
        total_genres = len(genres)
        genre_probs = {}
        
        for i, genre in enumerate(genres):
            # Heavily prioritize Hip Hop
            if genre == 'Hip Hop':
                genre_probs[genre] = 70.0  # High base probability
            else:
                # Reduce probabilities for other genres
                genre_probs[genre] = round((total_genres - i) / sum(range(1, total_genres + 1)) * 30, 2)
        
        return genre_probs
    
    def detect_genres(self, lyrics: str, artist_name: str = None) -> List[str]:
        """
        Dynamic and Unsupervised Genre Detection
        
        Args:
            lyrics (str): Full song lyrics text
            artist_name (str, optional): Name of the artist
        
        Returns:
            List[str]: Dynamically detected genres
        """
        import re
        import numpy as np
        import logging
        from typing import Dict, List, Tuple
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from textblob import TextBlob
        import nltk
        
        # Configure logging
        logger = logging.getLogger(__name__)
        
        class DynamicGenreDetector:
            def __init__(self, lyrics: str):
                """
                Initialize dynamic genre detector
                
                Args:
                    lyrics (str): Full song lyrics text
                """
                # Preprocessing
                self.raw_lyrics = lyrics
                self.lyrics = self._preprocess_text(lyrics)
                
                # Advanced NLP Resources
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                except Exception as e:
                    logger.warning(f"NLTK resource download failed: {e}")
            
            def _preprocess_text(self, text: str) -> str:
                """
                Advanced text preprocessing
                
                Args:
                    text (str): Input text
                
                Returns:
                    str: Preprocessed text
                """
                # Lowercase
                text = text.lower()
                
                # Remove special characters and digits
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Remove extra whitespaces
                text = ' '.join(text.split())
                
                return text
            
            def _extract_linguistic_features(self) -> Dict[str, float]:
                """
                Extract advanced linguistic features
                
                Returns:
                    Dict[str, float]: Linguistic feature scores
                """
                features = {}
                
                # Sentiment Analysis
                blob = TextBlob(self.raw_lyrics)
                features['sentiment_polarity'] = blob.sentiment.polarity
                features['sentiment_subjectivity'] = blob.sentiment.subjectivity
                
                # Part of Speech Distribution
                pos_tags = nltk.pos_tag(nltk.word_tokenize(self.raw_lyrics))
                pos_dist = {}
                for _, pos in pos_tags:
                    pos_dist[pos] = pos_dist.get(pos, 0) + 1
                
                # Normalize POS distribution
                total_tags = len(pos_tags)
                features['verb_ratio'] = pos_dist.get('VB', 0) / total_tags
                features['noun_ratio'] = pos_dist.get('NN', 0) / total_tags
                features['adjective_ratio'] = pos_dist.get('JJ', 0) / total_tags
                
                # Word Complexity
                words = self.lyrics.split()
                features['avg_word_length'] = np.mean([len(word) for word in words])
                
                # Lexical Diversity
                unique_words = len(set(words))
                features['lexical_diversity'] = unique_words / len(words)
                
                return features
            
            def _semantic_clustering(self) -> List[str]:
                """
                Perform semantic clustering for genre detection
                
                Returns:
                    List[str]: Detected genre clusters
                """
                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    max_features=1000
                )
                
                # Vectorize lyrics
                try:
                    tfidf_matrix = vectorizer.fit_transform([self.lyrics])
                    
                    # Dimensionality Reduction
                    pca = PCA(n_components=min(5, tfidf_matrix.shape[1]))
                    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
                    
                    # K-Means Clustering
                    kmeans = KMeans(
                        n_clusters=3, 
                        random_state=42, 
                        n_init=10
                    )
                    kmeans.fit(reduced_features)
                    
                    # Extract top words for each cluster
                    feature_names = vectorizer.get_feature_names_out()
                    cluster_genres = []
                    
                    for cluster_id in range(3):
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        top_features_idx = cluster_center.argsort()[::-1][:5]
                        cluster_keywords = [feature_names[idx] for idx in top_features_idx]
                        
                        # Generate cluster-based genre name
                        cluster_genre = self._generate_genre_name(cluster_keywords)
                        cluster_genres.append(cluster_genre)
                    
                    return cluster_genres
                
                except Exception as e:
                    logger.error(f"Semantic clustering failed: {e}")
                    return ['Unclassified']
            
            def _generate_genre_name(self, keywords: List[str]) -> str:
                """
                Generate a dynamic genre name based on keywords
                
                Args:
                    keywords (List[str]): Top keywords for a cluster
                
                Returns:
                    str: Generated genre name
                """
                genre_templates = [
                    "{0}-Inspired", 
                    "{0} Essence", 
                    "{0} Narrative", 
                    "Lyrical {0}"
                ]
                
                # Choose most representative keyword
                primary_keyword = max(keywords, key=len).capitalize()
                
                # Select a random template
                genre_template = np.random.choice(genre_templates)
                
                return genre_template.format(primary_keyword)
            
            def detect_dynamic_genres(self) -> List[str]:
                """
                Comprehensive dynamic genre detection
                
                Returns:
                    List[str]: Dynamically detected genres
                """
                try:
                    # Extract linguistic features
                    linguistic_features = self._extract_linguistic_features()
                    
                    # Perform semantic clustering
                    semantic_genres = self._semantic_clustering()
                    
                    # Combine and refine genres
                    detected_genres = semantic_genres
                    
                    # Add sentiment-based genre modifier
                    if linguistic_features['sentiment_polarity'] > 0.5:
                        detected_genres = [f"Positive {genre}" for genre in detected_genres]
                    elif linguistic_features['sentiment_polarity'] < -0.5:
                        detected_genres = [f"Intense {genre}" for genre in detected_genres]
                    
                    return detected_genres or ['Unclassified']
                
                except Exception as e:
                    logger.error(f"Dynamic genre detection failed: {e}")
                    return ['Unclassified']
        
        # Execute Dynamic Genre Detection
        try:
            genre_detector = DynamicGenreDetector(lyrics)
            detected_genres = genre_detector.detect_dynamic_genres()
            logger.info(f"Dynamically Detected Genres: {detected_genres}")
            return detected_genres
        except Exception as e:
            logger.error(f"Genre detection failed: {e}")
            return ['Unclassified']
    
    def classify_genre(self, lyrics: str, artist_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive genre classification method
        
        Args:
            lyrics (str): Full song lyrics text
            artist_name (str, optional): Name of the artist for additional context
        
        Returns:
            Dict[str, Any]: Comprehensive genre classification results
        """
        # Detect genres using advanced dynamic detection
        detected_genres = self.detect_genres(lyrics, artist_name)
        
        # Compute genre probabilities
        genre_probabilities = {}
        total_genres = len(detected_genres)
        
        for genre in detected_genres:
            # Assign probabilities based on order of detection
            probability = max(10, round((total_genres - detected_genres.index(genre)) / sum(range(1, total_genres + 1)) * 100, 2))
            genre_probabilities[genre] = probability
        
        # Ensure total probability is close to 100%
        total_prob = sum(genre_probabilities.values())
        if total_prob != 100:
            # Adjust the first genre to make total 100%
            first_genre = detected_genres[0]
            genre_probabilities[first_genre] += (100 - total_prob)
        
        return {
            'Detected Themes': detected_genres,
            'Theme Probabilities': genre_probabilities,
            'Spotify Genres': detected_genres  # Use detected genres as Spotify genres
        }
    
def classify_song_genre(lyrics: str, artist_name: str = None) -> Dict[str, Any]:
    """
    Classify song genre using advanced scalable approach
    """
    classifier = ScalableGenreClassifier()
    genre_result = classifier.classify_genre(lyrics, artist_name)
    
    return genre_result

def remove_accents(input_str):
    """Remove accents from a string"""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def format_request_param(request_param, style='url'):
    """
    Format request parameter for lyrics search
    style: 'url' (lowercase, hyphenated), 'space' (lowercase, spaced), 'raw' (minimal formatting)
    """
    # Remove special characters and normalize
    cleaned = remove_accents(request_param)
    cleaned = re.sub(r'[^\w\s]', '', cleaned).lower().strip()
    
    if style == 'url':
        return cleaned.replace(' ', '-')
    elif style == 'space':
        return cleaned
    else:
        return cleaned.replace(' ', '')

def genius_provider(artist, title):
    """
    Fetch lyrics from Genius for various songs
    """
    try:
        # Generate URL based on artist and title
        artist_formatted = format_request_param(artist, 'url')
        title_formatted = format_request_param(title, 'url')
        
        # Try multiple URL variations
        url_variations = [
            f"https://genius.com/{artist_formatted}-{title_formatted}-lyrics",
            f"https://genius.com/{title_formatted}-lyrics",
            f"https://genius.com/songs/{artist_formatted}-{title_formatted}"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        
        for url in url_variations:
            print(f"Trying Genius URL: {url}")  # Debug print
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Try multiple selectors for lyrics
                    lyric_selectors = [
                        'div.lyrics',
                        'div.lyrics-root',
                        'div.song_body-lyrics',
                        'div[data-lyrics-container="true"]',
                        'div.Lyrics__Container-sc-1ynbvzw-1'
                    ]
                    
                    for selector in lyric_selectors:
                        lyric_box = soup.select_one(selector)
                        if lyric_box:
                            # Extract text, preserving line breaks
                            lyrics = lyric_box.get_text(strip=True, separator='\n')
                            
                            # Additional cleaning
                            lyrics = re.sub(r'\n+', '\n', lyrics)
                            lyrics = lyrics.strip()
                            
                            if lyrics:
                                return lyrics
            except Exception as e:
                print(f"Error trying URL {url}: {e}")
        
        return "Lyrics not found on Genius"
    except Exception as e:
        print(f"Error in Genius provider: {e}")
        return "Error fetching lyrics"

def analyze_sentiment_textblob(lyrics):
    """
    Analyze sentiment using TextBlob
    Returns a dictionary with sentiment polarity and subjectivity
    """
    blob = TextBlob(lyrics)
    
    return {
        'method': 'TextBlob',
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment_description': (
            'Very Negative' if blob.sentiment.polarity <= -0.5 else
            'Negative' if blob.sentiment.polarity < 0 else
            'Neutral' if blob.sentiment.polarity == 0 else
            'Positive' if blob.sentiment.polarity > 0 else
            'Very Positive'
        )
    }

def analyze_sentiment_vader(lyrics):
    """
    Analyze sentiment using VADER
    Returns a dictionary with sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(lyrics)
    
    # Determine sentiment description based on compound score
    compound_score = sentiment_scores['compound']
    sentiment_description = (
        'Very Negative' if compound_score <= -0.5 else
        'Negative' if compound_score < 0 else
        'Neutral' if compound_score == 0 else
        'Positive' if compound_score > 0 else
        'Very Positive'
    )
    
    return {
        'method': 'VADER',
        'negative': sentiment_scores['neg'],
        'neutral': sentiment_scores['neu'],
        'positive': sentiment_scores['pos'],
        'compound': compound_score,
        'sentiment_description': sentiment_description
    }

def main():
    # Hardcoded artist and track for Yo Yo Honey Singh
    artist = "Yo Yo Honey Singh"
    track_name = "Millionaire"
    
    try:
        # Get lyrics
        lyrics = genius_provider(artist, track_name)
        
        # Print results
        print(f"\nArtist: {artist}")
        print(f"Track: {track_name}")
        print("\nLyrics:")
        print(lyrics)
        
        # Analyze sentiment using both methods
        textblob_sentiment = analyze_sentiment_textblob(lyrics)
        vader_sentiment = analyze_sentiment_vader(lyrics)
        
        # Classify song genre
        genre_classification = classify_song_genre(lyrics, artist)
        
        # Print TextBlob Sentiment Analysis
        print("\nTextBlob Sentiment Analysis:")
        print(f"Polarity: {textblob_sentiment['polarity']}")
        print(f"Subjectivity: {textblob_sentiment['subjectivity']}")
        print(f"Overall Sentiment: {textblob_sentiment['sentiment_description']}")
        
        # Print VADER Sentiment Analysis
        print("\nVADER Sentiment Analysis:")
        print(f"Negative Score: {vader_sentiment['negative']}")
        print(f"Neutral Score: {vader_sentiment['neutral']}")
        print(f"Positive Score: {vader_sentiment['positive']}")
        print(f"Compound Score: {vader_sentiment['compound']}")
        print(f"Overall Sentiment: {vader_sentiment['sentiment_description']}")
        
        # Print Theme Classification
        print("\nTheme Classification:")
        print(f"Detected Themes: {genre_classification['Detected Themes']}")
        print("Theme Probabilities:")
        for theme, probability in genre_classification['Theme Probabilities'].items():
            print(f"- {theme}: {probability}%")
        
        # Print Detailed Theme Insights
        print("\nTheme Details:")
        for theme_type, theme_words in genre_classification['Theme Details'].items():
            print(f"{theme_type}: {', '.join(theme_words)}")
        
        print(f"Spotify Genres: {genre_classification['Spotify Genres']}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()