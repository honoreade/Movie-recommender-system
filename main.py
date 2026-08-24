import os
import re
import difflib
import threading
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from functools import lru_cache

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# --- Constants & Configuration ---
CSV_FILE_PATH = '25k IMDb movie Dataset.csv'
DEFAULT_MODEL_KEY   = 'weighted'

MODEL_ORDER = ['weighted', 'plot', 'genres', 'director']
MODEL_LABELS = {
    'weighted': '✨ Combined Recommendations',
    'plot': '📖 Plot similarity',
    'genres': '🎭 Genre similarity',
    'director': '🎬 Same Director'
}

# --- Globals for caching models ---
MODEL_CATALOG = {}
MODEL_LOCK = threading.Lock()
_SHARED_MOVIES = None
_SHARED_METADATA = None
_SHARED_LOCK = threading.Lock()


# --- Helper to load shared dataset ---
def get_shared_movies():
    global _SHARED_MOVIES, _SHARED_METADATA
    with _SHARED_LOCK:
        if _SHARED_MOVIES is not None:
            return _SHARED_MOVIES

        print(f"Loading shared movie dataset from {CSV_FILE_PATH} ...")
        movies = pd.read_csv(CSV_FILE_PATH)
        movies = movies.dropna(subset=['movie title'])

        title_lookup = {}
        for original_title in movies['movie title'].unique():
            norm = str(original_title).lower().strip()
            title_lookup[norm] = original_title

        title_keys = list(title_lookup.keys())
        title_to_index = {str(row['movie title']).lower().strip(): idx for idx, row in movies.iterrows()}

        _SHARED_MOVIES = (movies, title_lookup, title_keys, title_to_index)
        _SHARED_METADATA = movies.drop_duplicates(subset=['movie title']).set_index('movie title').to_dict('index')
        return _SHARED_MOVIES

# --- Fetch Metadata ---
def get_movie_metadata(title):
    _, _, _, _ = get_shared_movies()
    if title not in _SHARED_METADATA:
        return {'title': title, 'year': '', 'rating': '', 'genres': '', 'director': '', 'overview': '', 'path': ''}
    
    row = _SHARED_METADATA[title]
    year_raw = str(row.get('year', ''))
    year = year_raw.replace('-', '') if year_raw and year_raw != 'nan' else ''
    
    return {
        'title': title,
        'year': year,
        'rating': str(row.get('Rating', '')) if str(row.get('Rating', '')) != 'nan' else '',
        'genres': str(row.get('Generes', '')).replace("['", "").replace("']", "").replace("', '", ", ") if str(row.get('Generes', '')) != 'nan' else '',
        'director': str(row.get('Director', '')) if str(row.get('Director', '')) != 'nan' else '',
        'overview': str(row.get('Overview', '')) if str(row.get('Overview', '')) != 'nan' else '',
        'path': str(row.get('path', '')) if str(row.get('path', '')) != 'nan' else ''
    }

def fetch_imdb_poster_url(imdb_path):
    if not imdb_path:
        return ""
    m = re.search(r'/title/(tt\d+)', imdb_path)
    if m:
        imdb_id = m.group(1)
        return f"https://images.metahub.space/poster/small/{imdb_id}/img"
    return ""


from search import find_best_movie_match, find_top_movie_matches

# --- Model Builders ---
def build_feature_matrix(movies, model_key):
    if model_key == 'weighted':
        features = movies['Generes'].fillna('') + " " + movies['Plot Kyeword'].fillna('') + " " + movies['Director'].fillna('') + " " + movies['Top 5 Casts'].fillna('') + " " + movies['Overview'].fillna('')
    elif model_key == 'plot':
        features = movies['Overview'].fillna('') + " " + movies['Plot Kyeword'].fillna('')
    elif model_key == 'genres':
        features = movies['Generes'].fillna('')
    elif model_key == 'director':
        features = movies['Director'].fillna('')
    else:
        features = movies['Generes'].fillna('')
    return features


def prepare_artifact_cache(model_key):
    with MODEL_LOCK:
        if model_key in MODEL_CATALOG:
            return MODEL_CATALOG[model_key]

        movies, title_lookup, title_keys, title_to_index = get_shared_movies()
        features = build_feature_matrix(movies, model_key)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        feature_matrix = vectorizer.fit_transform(features)

        nn = NearestNeighbors(metric='cosine', algorithm='brute')
        nn.fit(feature_matrix)

        engine = {
            'movies': movies,
            'title_lookup': title_lookup,
            'title_keys': title_keys,
            'title_to_index': title_to_index,
            'feature_matrix': feature_matrix,
            'vectorizer': vectorizer,
            'nn': nn,
            'top_n': 10,
        }
        MODEL_CATALOG[model_key] = engine
        return engine


@lru_cache(maxsize=1024)
def recommend_for_artifact(model_key, query, top_n=8):
    try:
        engine = prepare_artifact_cache(model_key)
        if engine is None:
            return {'match': None, 'match_meta': None, 'score': None, 'suggestions': [], 'recommendations': []}

        movies = engine['movies']
        title_lookup = engine['title_lookup']
        title_keys = engine['title_keys']
        title_to_index = engine['title_to_index']
        feature_matrix = engine['feature_matrix']
        nn = engine['nn']

        matched_title = find_best_movie_match(query, title_keys, title_lookup)
        suggestions = [matched_title] if matched_title else []
        match_meta = get_movie_metadata(matched_title) if matched_title else None

        if suggestions:
            matched_index = title_to_index.get(str(matched_title).lower().strip())
            if matched_index is not None:
                query_vector = feature_matrix[matched_index:matched_index + 1]
                distances, indices = nn.kneighbors(query_vector, n_neighbors=min(top_n + 1, len(movies)))
                
                recommendations = []
                seen_titles = {str(matched_title).lower().strip()}
                for idx in indices[0]:
                    movie_title_raw = str(movies.iloc[idx]['movie title'])
                    movie_title_clean = movie_title_raw.lower().strip()
                    if movie_title_clean in seen_titles:
                        continue
                    seen_titles.add(movie_title_clean)
                    recommendations.append(get_movie_metadata(movie_title_raw))
                    if len(recommendations) >= top_n:
                        break

                score = round(float(distances[0][0]), 4) if len(distances) and len(distances[0]) else None
                return {
                    'match': matched_title,
                    'match_meta': match_meta,
                    'score': score,
                    'suggestions': suggestions,
                    'recommendations': recommendations,
                }

        if engine.get('vectorizer') is not None:
            query_vector = engine['vectorizer'].transform([query])
            distances, indices = nn.kneighbors(query_vector, n_neighbors=min(top_n, len(movies)))
            recommendations = [get_movie_metadata(str(movies.iloc[idx]['movie title'])) for idx in indices[0]][:top_n]
            score = round(float(distances[0][0]), 4) if len(distances) and len(distances[0]) else None
            return {
                'match': None,
                'match_meta': None,
                'score': score,
                'suggestions': suggestions,
                'recommendations': recommendations,
            }

        return {'match': None, 'match_meta': None, 'score': None, 'suggestions': suggestions, 'recommendations': []}

    except Exception as exc:
        print(f"Error computing recommendations for model '{model_key}': {exc}")
        return {'match': None, 'match_meta': None, 'score': None, 'suggestions': [], 'recommendations': [], 'error': str(exc)}


# --- Flask Routes ---
@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    movies_df, title_lookup, title_keys, title_to_index = get_shared_movies()
    top_matches = find_top_movie_matches(query, title_keys, title_lookup, n=5)
    return jsonify(top_matches)

@app.route('/api/poster')
def api_poster():
    imdb_path = request.args.get('path', '').strip()
    title = request.args.get('title', '').strip()
    if not imdb_path and title:
        meta = get_movie_metadata(title)
        imdb_path = meta.get('path', '')

    if not imdb_path:
        return jsonify({'poster_url': '', 'path': ''})

    poster_url = fetch_imdb_poster_url(imdb_path)
    return jsonify({'poster_url': poster_url, 'path': imdb_path})

@app.route('/model-result/<model_key>')
def model_result(model_key):
    movie_name = request.args.get('mname', '').strip()
    top_n = request.args.get('top_n', 8, type=int)
    if not movie_name:
        return jsonify({
            'name': MODEL_LABELS.get(model_key, model_key),
            'payload': {'match': None, 'match_meta': None, 'score': None, 'suggestions': [], 'recommendations': []}
        })
    payload = recommend_for_artifact(model_key, movie_name, top_n=top_n)
    return jsonify({
        'name': MODEL_LABELS.get(model_key, model_key),
        'payload': payload,
    })

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        movie_name = request.form.get('mname', '').strip()
    else:
        movie_name = request.args.get('mname', '').strip()

    if not movie_name:
        return redirect(url_for('index'))
    return redirect(url_for('index', q=movie_name))

@app.route('/')
def index():
    query = request.args.get('q', '').strip()
    matched_title = None
    not_found = False

    if query:
        movies, title_lookup, title_keys, title_to_index = get_shared_movies()
        matched_title = find_best_movie_match(query, title_keys, title_lookup)
        if not matched_title:
            not_found = True

    return render_template(
        'index.html',
        query=query,
        matched_title=matched_title,
        not_found=not_found,
        model_key=DEFAULT_MODEL_KEY,
        model_order=MODEL_ORDER,
        model_labels=MODEL_LABELS,
        results={},
    )

def warmup_all_models():
    print("Pre-loading models, metadata, and vector caches in background...")
    try:
        get_movie_metadata("Inception")
        print("  [OK] Movie metadata loaded.")
    except Exception as exc:
        print(f"  [WARN] Failed to preload metadata: {exc}")

    for key in MODEL_ORDER:
        try:
            prepare_artifact_cache(key)
            print(f"  [OK] Model '{key}' loaded and ready.")
        except Exception as exc:
            print(f"  [WARN] Failed to preload '{key}': {exc}")
    print("All models ready in memory!")

if __name__ == '__main__':
    threading.Thread(target=warmup_all_models, daemon=True).start()
    print('Starting Flask application on http://127.0.0.1:7500 ...')
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=7500)
    