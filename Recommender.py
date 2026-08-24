import difflib
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
CSV_PATH = BASE_DIR / '25k IMDb movie Dataset.csv'

# ==============================================================================
# Model Artifact Registry
# ==============================================================================
ARTIFACT_MODELS = {
    'weighted': {
        'name': 'Weighted TF-IDF (Default)',
        'joblib': 'weighted_tfidf_recommender.joblib',
        'feature_file': 'weighted_tfidf_features.npz',
        'type': 'sparse',
    },
    'all_minilm': {
        'name': 'all-MiniLM-L6-v2 Semantic Embeddings',
        'joblib': 'all_minilm_recommender.joblib',
        'feature_file': 'all_minilm_vectors.npy',
        'type': 'dense',
    },
    'tfidf': {
        'name': 'Standard TF-IDF',
        'joblib': 'tfidf_recommender.joblib',
        'feature_file': 'tfidf_features.npz',
        'type': 'sparse',
    },
    'word2vec': {
        'name': 'Word2Vec Dense Vectors',
        'joblib': 'word2vec_recommender.joblib',
        'feature_file': 'word2vec_vectors.npy',
        'type': 'dense',
    },
    'fasttext': {
        'name': 'FastText Subword Embeddings',
        'joblib': 'fasttext_recommender.joblib',
        'feature_file': 'fasttext_vectors.npy',
        'type': 'dense',
    },
}

_SHARED_MOVIES = None
_MODEL_CACHE = {}


def get_shared_movies():
    """Lazy-loads and caches the movie DataFrame and title lookups."""
    global _SHARED_MOVIES
    if _SHARED_MOVIES is not None:
        return _SHARED_MOVIES

    # 1. Try loading from any available joblib artifact
    for model_info in ARTIFACT_MODELS.values():
        joblib_path = ARTIFACTS_DIR / model_info['joblib']
        if joblib_path.exists():
            try:
                data = joblib.load(joblib_path)
                df = data.get('movies')
                if df is not None and not df.empty:
                    _SHARED_MOVIES = df.reset_index(drop=True)
                    return _SHARED_MOVIES
            except Exception:
                continue

    # 2. Fallback to reading the raw CSV dataset
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        _SHARED_MOVIES = df.reset_index(drop=True)
        return _SHARED_MOVIES

    raise FileNotFoundError("Neither joblib artifacts nor dataset CSV found.")


def load_model_engine(model_key: str = 'weighted'):
    """
    Loads and caches the feature matrix and NearestNeighbors index for a specific model.
    Falls back to building TF-IDF on-the-fly if artifact is missing.
    """
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    movies = get_shared_movies()
    model_info = ARTIFACT_MODELS.get(model_key)
    feature_matrix = None

    if model_info:
        feature_path = ARTIFACTS_DIR / model_info['feature_file']
        if feature_path.exists():
            try:
                if model_info['type'] == 'sparse':
                    npz = np.load(feature_path)
                    feature_matrix = sp.csr_matrix(
                        (npz['data'], npz['indices'], npz['indptr']),
                        shape=npz['shape']
                    )
                else:
                    feature_matrix = np.load(feature_path)
            except Exception as exc:
                print(f"[Warning] Failed loading {feature_path.name}: {exc}")

    # Fallback to computing TF-IDF directly from CSV if artifact is missing
    if feature_matrix is None:
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            for col in ['movie title', 'Generes', 'Overview', 'Plot Kyeword', 'Director', 'Top 5 Casts', 'Writer']:
                if col not in df.columns:
                    df[col] = ''
                else:
                    df[col] = df[col].fillna('')

            features_text = (
                df['movie title'] + ' ' +
                df['Generes'] + ' ' +
                df['Overview'] + ' ' +
                df['Plot Kyeword'] + ' ' +
                df['Director'] + ' ' +
                df['Top 5 Casts'] + ' ' +
                df['Writer']
            )
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
            feature_matrix = vectorizer.fit_transform(features_text.values.astype('U'))
        else:
            raise RuntimeError(f"Cannot initialize model '{model_key}'.")

    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(feature_matrix)

    engine = {
        'movies': movies,
        'feature_matrix': feature_matrix,
        'nn': nn,
    }
    _MODEL_CACHE[model_key] = engine
    return engine


def get_recommendations(
    movie_name: str,
    model: str = 'weighted',
    top_n: int = 30,
    callback: Optional[Callable[[str, str, List[str]], None]] = None
) -> List[str]:
    """
    Computes movie recommendations using the requested model artifact.
    
    If a callback is provided, it is invoked with:
        callback(model_key, model_name, recommended_titles)

    Returns:
        List of recommended movie title strings only.
    """
    if model not in ARTIFACT_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(ARTIFACT_MODELS.keys())}")

    engine = load_model_engine(model)
    movies = engine['movies']
    features = engine['feature_matrix']
    nn = engine['nn']

    list_of_all_titles = movies['movie title'].dropna().astype(str).tolist()
    title_lookup = {t.lower(): t for t in list_of_all_titles}
    query_clean = str(movie_name).strip().lower()

    if not query_clean:
        if callback:
            callback(model, ARTIFACT_MODELS[model]['name'], [])
        return []

    # Title resolution: Exact match first, then fuzzy match
    if query_clean in title_lookup:
        close_match = title_lookup[query_clean]
    else:
        find_close_match = difflib.get_close_matches(query_clean, [t.lower() for t in list_of_all_titles], n=1, cutoff=0.6)
        if not find_close_match:
            if callback:
                callback(model, ARTIFACT_MODELS[model]['name'], [])
            return []
        close_match = title_lookup[find_close_match[0]]

    # Movie index lookup
    index_matches = movies.index[movies['movie title'] == close_match].tolist()
    if not index_matches:
        if callback:
            callback(model, ARTIFACT_MODELS[model]['name'], [])
        return []
    index_of_the_movie = index_matches[0]

    # Nearest neighbor vector search
    query_vector = features[index_of_the_movie:index_of_the_movie + 1]
    distances, indices = nn.kneighbors(query_vector, n_neighbors=min(top_n + 1, len(movies)))

    recommended_titles = []
    for idx in indices[0]:
        if idx == index_of_the_movie:
            continue
        title = str(movies.iloc[idx]['movie title']).strip()
        recommended_titles.append(title)
        if len(recommended_titles) >= top_n:
            break

    # Trigger callback if provided
    if callback is not None:
        callback(model, ARTIFACT_MODELS[model]['name'], recommended_titles)

    return recommended_titles


def run_all_models_with_callback(
    movie_name: str,
    top_n: int = 10,
    callback: Optional[Callable[[str, str, List[str]], None]] = None
) -> Dict[str, List[str]]:
    """
    Executes recommendation across all available model artifacts and feeds
    results to the provided callback function as each model completes.
    """
    results = {}
    for model_key in ARTIFACT_MODELS:
        try:
            titles = get_recommendations(movie_name, model=model_key, top_n=top_n, callback=callback)
            results[model_key] = titles
        except Exception as exc:
            print(f"[Error in {model_key}]: {exc}")
            results[model_key] = []
    return results


# ==============================================================================
# Example Callbacks for Execution
# ==============================================================================
def title_display_callback(model_key: str, model_name: str, titles: List[str]):
    """Default callback that prints recommended movie titles only."""
    print(f"\n--- {model_name} ---")
    if not titles:
        print("  No recommendations found.")
        return
    for i, title in enumerate(titles, start=1):
        print(f"{i} . {title}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == '__main__':
    # 1. Parse movie query from CLI arguments or input prompt
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        movie_name = ' '.join(sys.argv[1:])
    else:
        try:
            movie_name = input('Enter your favourite movie name : ').strip()
        except (EOFError, KeyboardInterrupt):
            movie_name = 'avatar'

    if not movie_name:
        movie_name = 'avatar'

    # 2. Check if user wants to run single model or compare all via callbacks
    if '--all' in sys.argv:
        print(f"\nRunning all model artifacts for '{movie_name}' with callbacks:\n" + "=" * 50)
        run_all_models_with_callback(movie_name, top_n=10, callback=title_display_callback)
    else:
        # Default: Run Weighted TF-IDF artifact
        titles = get_recommendations(movie_name, model='weighted', top_n=30, callback=title_display_callback)

# ==============================================================================
# NOTE ON GITHUB ARTIFACT PUSH, CLI SUPPORT & CALLBACK EXECUTION:
# ==============================================================================
# 1. CLI Support:
#    - Run default model:
#      python Recommender.py "Avatar"
#    - Run all models via callbacks:
#      python Recommender.py "Avatar" --all
#
# 2. GitHub Artifact Tracking:
#    - Only 'artifacts/weighted_tfidf_*' is tracked and pushed to GitHub because
#      it is compact (<50MB) and meets GitHub's 100MB file size limit.
#    - Other model artifacts ('all_minilm' ~140MB, 'fasttext' ~926MB, 'word2vec' ~83MB)
#      are listed in .gitignore (which remains untouched) and can be generated
#      locally via Recommender-note-new.ipynb.
#
# 3. Callback-Driven Execution Pattern:
#    - You can execute individual or multiple models asynchronously/sequentially
#      by passing a custom callback function:
#
#      def my_callback(model_key, model_name, titles):
#          # Consume or dispatch movie titles only
#          print(f"Model {model_name} generated {len(titles)} titles.")
#          for t in titles:
#              print(" -", t)
#
#      # Single model callback:
#      get_recommendations("Avatar", model="weighted", top_n=10, callback=my_callback)
#
#      # Multi-model batch callback:
#      run_all_models_with_callback("Inception", top_n=5, callback=my_callback)
# ==============================================================================
