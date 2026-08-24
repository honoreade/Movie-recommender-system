import difflib

def find_best_movie_match(query, title_keys, title_lookup):
    normalized_q = query.lower().strip()
    if not normalized_q:
        return None
        
    if normalized_q in title_lookup:
        return title_lookup[normalized_q]
        
    word_matches = [t for t in title_keys if normalized_q in t.split() or f" {normalized_q} " in f" {t} "]
    if word_matches:
        word_matches.sort(key=len)
        return title_lookup[word_matches[0]]
        
    prefix_matches = [t for t in title_keys if t.startswith(normalized_q) or f" {normalized_q}" in t]
    if prefix_matches:
        prefix_matches.sort(key=len)
        return title_lookup[prefix_matches[0]]
        
    matches = difflib.get_close_matches(normalized_q, title_keys, n=1, cutoff=0.90)
    if matches:
        return title_lookup[matches[0]]
        
    substring_matches = [t for t in title_keys if normalized_q in t]
    if substring_matches:
        substring_matches.sort(key=len)
        return title_lookup[substring_matches[0]]
        
    return None

def find_top_movie_matches(query, title_keys, title_lookup, n=5):
    normalized_q = query.lower().strip()
    if not normalized_q:
        return []
        
    results = []
    seen = set()
    
    def add_match(t_key):
        if t_key not in seen:
            seen.add(t_key)
            results.append(title_lookup[t_key])
            
    if normalized_q in title_lookup:
        add_match(normalized_q)
        
    word_matches = [t for t in title_keys if normalized_q in t.split() or f" {normalized_q} " in f" {t} "]
    word_matches.sort(key=len)
    for t in word_matches: add_match(t)
    if len(results) >= n: return results[:n]
        
    prefix_matches = [t for t in title_keys if t.startswith(normalized_q) or f" {normalized_q}" in t]
    prefix_matches.sort(key=len)
    for t in prefix_matches: add_match(t)
    if len(results) >= n: return results[:n]
        
    matches = difflib.get_close_matches(normalized_q, title_keys, n=n, cutoff=0.90)
    for t in matches: add_match(t)
    if len(results) >= n: return results[:n]
        
    substring_matches = [t for t in title_keys if normalized_q in t]
    substring_matches.sort(key=len)
    for t in substring_matches: add_match(t)
    
    return results[:n]
