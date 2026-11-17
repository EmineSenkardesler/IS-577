import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    # Handle non-string or missing values
    if not isinstance(text, str) or pd.isna(text):
        return ''
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

def load_and_preprocess_data(df):
    """Load data and preprocess the Summary column."""
    # Verify that 'Summary' column exists
    if 'Summary' not in df.columns:
        raise ValueError("Dataset does not contain a 'Summary' column")
    
    # Apply preprocessing to Summary column
    df['Processed_Summary'] = df['Summary'].apply(preprocess_text)
    
    # Define categories
    categories = [
        "WEATHER", "OCEAN", "LAND", "SAFETY", "WATER", "SNOW", "HEALTH", "TRANSPORTATION",
        "DIEOFF", "EXTREME_TEMP", "FISH", "LAND_ANIMALS", "INFRASTRUCTURE", "SEA_ANIMALS",
        "PLANTS", "FOODSECURITY", "BIRD", "RANGE_CHANGE", "SEASON_TIMING", "EXTREME_PRECIP",
        "ECONOMICIMPACT", "RECREATION", "INVERTEBRATES", "CULTURALIMPACT", "FLOOD", "INSECT",
        "AIR", "ENERGY", "PERMAFROST", "WILDFIRE", "EXTREME_WIND", "DEFORMITY", "ABUNDANCE",
        "FISHERIES", "SANITATION", "BEHAVIOR", "EROSION", "LANDSLIDE", "POLLUTION",
        "WATERSECURITY", "DROUGHT", "MICROBES", "VIRULENCE", "HARVEST_FAILURE", "AIRQUALITY",
        "PETS", "AQUA_AG", "DISPLACEMENT", "LIVESTOCK", "PARASITE", "ALGAL_BLOOM",
        "WATER_MANAGEMENT", "ONEHEALTH", "ZOONOTIC_FLU", "ROS", "INFESTATION", "ARACHNIDS",
        "STORMSURGE", "GLACIER_CHANGE", "ATMOSPHERE", "GROUNDWATER", "TOURISM",
        "COMMUNICATION", "DEBRIS", "TORNADO", "PARALYTIC_SP", "EARTHQUAKE", "FORESTRY",
        "AQUACULTURE", "VOLCANO", "THUNDERSTORM", "FUNGI", "REPTILES", "COLD_BLOODS",
        "CONFLICT", "CHEMICAL_CHANGE", "INJURY", "SEA_LEVEL_CHANGE", "ACUTE_ILLNESS",
        "MENTAL_HEALTH", "BURIAL_SITE", "RABIES", "SINKHOLE", "ANTHRAX", "RESILIENCE",
        "DISTEMPER", "RELOCATION", "TSUNAMI", "CWD", "CHRONIC_ILLNESS", "TULAREMIA",
        "VIBRIO", "UNUSUAL_VARIATION", "AMNESIC_SP", "TRICHINOSIS", "TOXOPLASMOSIS",
        "RADIATION", "OTHER", "BOTULISM", "ALASKAPOX", "BRUCELLOSIS", "CORONAVIRUS",
        "ANISAKID", "QFEVER", "OTHER_EVENT", "GIARDIASIS", "SMELL_TASTE", "DIPHTHERIA",
        "MONKEYPOX"
    ]
    
    # Verify that category columns exist
    missing_categories = [cat for cat in categories if cat not in df.columns]
    if missing_categories:
        raise ValueError(f"Dataset is missing the following category columns: {missing_categories}")
    
    # Create labels: assume columns are binary (1 or 0) for each category
    df['Labels'] = df[categories].apply(
        lambda row: [cat for cat, val in zip(categories, row) if val == 1], axis=1
    )
    
    # Remove rows with empty Processed_Summary
    initial_rows = len(df)
    df = df[df['Processed_Summary'] != '']
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with empty Processed_Summary")
    
    return df, categories

def create_pipeline(df, categories):
    """Create and train the classification pipeline."""
    # Split data
    X = df['Processed_Summary']
    y = df['Labels']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Transform labels for multi-label classification
    mlb = MultiLabelBinarizer(classes=categories)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)
    
    # Train logistic regression with OneVsRestClassifier
    classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    classifier.fit(X_train_tfidf, y_train_bin)
    
    return classifier, vectorizer, mlb, X_test_tfidf, y_test_bin

def evaluate_model(classifier, X_test_tfidf, y_test_bin):
    """Evaluate the model using F1 score and Hamming loss."""
    y_pred = classifier.predict(X_test_tfidf)
    
    f1 = f1_score(y_test_bin, y_pred, average='micro')
    hamming = hamming_loss(y_test_bin, y_pred)
    
    print(f"Micro F1 Score: {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    
    return f1, hamming

def main():
    # Load dataset
    df = pd.read_csv('/Users/emine2/Documents/CLASSES/2025-SPRING/IS-577/project/leo_summary.csv')
    
    # Print column names for debugging
    print("Dataset columns:", df.columns.tolist())
    
    # Load and preprocess data
    df, categories = load_and_preprocess_data(df)
    
    # Create and train pipeline
    classifier, vectorizer, mlb, X_test_tfidf, y_test_bin = create_pipeline(df, categories)
    
    # Evaluate model
    evaluate_model(classifier, X_test_tfidf, y_test_bin)
    
    # Example: Predict on new data
    new_summary = ["Heavy rain caused flooding in coastal areas"]
    processed_summary = [preprocess_text(text) for text in new_summary]
    new_tfidf = vectorizer.transform(processed_summary)
    prediction = classifier.predict(new_tfidf)
    predicted_labels = mlb.inverse_transform(prediction)
    print(f"Predicted Labels: {predicted_labels}")

if __name__ == "__main__":
    main()