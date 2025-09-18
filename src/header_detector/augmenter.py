import os
import json
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from difflib import SequenceMatcher
from src.header_detector.config import HEADER_VOCAB_VARIATIONS_1
from typing import List, Tuple


OUTPUT_DIR = "./augmentation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.65
TOP_K = 20
TREES = 50  # Number of trees for Annoy index
ANNOY_SEARCH_K = -1  # -1 means search all trees, positive value for approximate search

model = SentenceTransformer(MODEL_NAME)


# -------------------------------
# Utility: Save JSON with description
# -------------------------------
def save_json(data, filename, description=""):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"// {description}\n")
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {description} -> {path}")


# -------------------------------
# Enhanced Annoy Wrapper with Advanced Sampling
# -------------------------------
class EnhancedAnnoyIndex:
    def __init__(self, dim: int, metric: str = 'angular', n_trees: int = 50):
        self.index = AnnoyIndex(dim, metric)
        self.dim = dim
        self.metric = metric
        self.n_trees = n_trees
        self.items = []
        self.is_built = False
    
    def add_item(self, i: int, vector: np.ndarray, item_data=None):
        """Add item to index with optional metadata"""
        self.index.add_item(i, vector)
        if len(self.items) <= i:
            self.items.extend([None] * (i - len(self.items) + 1))
        self.items[i] = item_data
    
    def build(self):
        """Build the index"""
        self.index.build(self.n_trees)
        self.is_built = True
        print(f"Built Annoy index with {self.index.get_n_items()} items and {self.n_trees} trees")
    
    def get_diverse_candidates(self, query_vector: np.ndarray, k: int, 
                              diversity_factor: float = 0.3) -> List[Tuple[int, float]]:
        """
        Get diverse top-k candidates using a combination of similarity and diversity sampling
        
        Args:
            query_vector: Query embedding vector
            k: Number of candidates to return
            diversity_factor: Weight for diversity (0.0 = pure similarity, 1.0 = pure diversity)
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        # Get initial larger candidate set
        initial_k = min(k * 3, self.index.get_n_items())
        candidates, distances = self.index.get_nns_by_vector(
            query_vector, initial_k, include_distances=True, search_k=ANNOY_SEARCH_K
        )
        
        if diversity_factor == 0.0 or len(candidates) <= k:
            # Pure similarity-based sampling or not enough candidates for diversity
            return [(candidates[i], 1 - (distances[i]**2)/2) for i in range(min(k, len(candidates)))]
        
        # Create candidate info with precomputed vectors and similarities
        candidate_info = []
        for i, (cand_idx, dist) in enumerate(zip(candidates, distances)):
            vec = np.array(self.index.get_item_vector(cand_idx))
            sim = 1 - (dist**2)/2
            candidate_info.append({
                'original_idx': cand_idx,
                'vector': vec,
                'similarity': sim,
                'position': i
            })
        
        # Select candidates using MMR
        selected = []
        remaining = candidate_info.copy()
        
        # Select first candidate (highest similarity)
        if remaining:
            best_candidate = remaining.pop(0)
            selected.append((best_candidate['original_idx'], best_candidate['similarity']))
        
        # Select remaining candidates using MMR
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                sim_to_query = candidate['similarity']
                
                # Maximum similarity to already selected items
                max_sim_to_selected = 0
                for sel_idx, _ in selected:
                    # Find selected candidate's vector
                    sel_vec = None
                    for sel_cand in candidate_info:
                        if sel_cand['original_idx'] == sel_idx:
                            sel_vec = sel_cand['vector']
                            break
                    
                    if sel_vec is not None:
                        dot_prod = np.dot(candidate['vector'], sel_vec)
                        max_sim_to_selected = max(max_sim_to_selected, dot_prod)
                
                # MMR score
                mmr_score = (1 - diversity_factor) * sim_to_query - diversity_factor * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = i
            
            # Add best candidate and remove from remaining
            if best_candidate is not None:
                selected.append((best_candidate['original_idx'], best_candidate['similarity']))
                remaining.pop(best_idx)
        
        return selected
    
    def sample_random_walk_candidates(self, query_vector: np.ndarray, k: int, 
                                     walk_length: int = 3) -> List[Tuple[int, float]]:
        """
        Sample candidates using random walks from nearest neighbors
        This explores the local neighborhood more thoroughly
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        # Start with top nearest neighbors
        initial_neighbors = self.index.get_nns_by_vector(
            query_vector, min(10, self.index.get_n_items()), 
            include_distances=False, search_k=ANNOY_SEARCH_K
        )
        
        visited = set()
        candidates = set()
        
        for start_node in initial_neighbors:
            current = start_node
            for step in range(walk_length):
                if current in visited:
                    break
                visited.add(current)
                candidates.add(current)
                
                # Get neighbors of current node
                current_vec = np.array(self.index.get_item_vector(current))
                neighbors = self.index.get_nns_by_vector(
                    current_vec, min(5, self.index.get_n_items()), 
                    include_distances=False, search_k=ANNOY_SEARCH_K
                )
                
                # Randomly select next node
                if neighbors:
                    current = random.choice(neighbors)
        
        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            candidate_vec = np.array(self.index.get_item_vector(candidate))
            similarity = np.dot(query_vector, candidate_vec)  # Assuming normalized vectors
            scored_candidates.append((candidate, similarity))
        
        # Sort by similarity and return top k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:k]
    
    def get_all_nns_in_radius(self, query_vector: np.ndarray, radius: float) -> List[Tuple[int, float]]:
        """
        Get all neighbors within a similarity radius
        This is useful for getting all relevant candidates rather than fixed k
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        # Start with a large k to get comprehensive results
        large_k = min(self.index.get_n_items(), 1000)
        candidates, distances = self.index.get_nns_by_vector(
            query_vector, large_k, include_distances=True, search_k=ANNOY_SEARCH_K
        )
        
        # Filter by radius (convert distance to similarity)
        filtered_candidates = []
        for idx, dist in zip(candidates, distances):
            similarity = 1 - (dist**2)/2  # Convert angular distance to cosine similarity
            if similarity >= radius:
                filtered_candidates.append((idx, similarity))
        
        return filtered_candidates


# -------------------------------
# Step 1: Programmatic Variant Generation
# -------------------------------
def gen_basic_variants(s):
    s = s.strip()
    variants = set([s, s.lower(), s.upper(), s.title()])
    
    # Punctuation variants
    variants.update([
        s.replace('.', ''), 
        s.replace(' ', ''), 
        s.replace(' ', '.'), 
        s.replace(' ', '/'),
        s.replace(' ', '_'),
        s.replace('_', ' '),
        s.replace('-', ' '),
        s.replace(' ', '-')
    ])
    
    # Character-drop variants
    if len(s) > 6:
        variants.add(s[:6])
        variants.add(s[-6:])  # Also try suffix
    
    # Abbreviations map
    abbr_map = {
        'amount': 'amt', 'transaction': 'txn', 'date': 'dt', 'reference': 'ref',
        'cheque': 'chq', 'withdrawal': 'wdl', 'deposit': 'dep', 'credit': 'cr',
        'debit': 'dr', 'balance': 'bal', 'number': 'no', 'description': 'desc'
    }
    
    s_lower = s.lower()
    for full, abbr in abbr_map.items():
        if full in s_lower:
            variants.add(s_lower.replace(full, abbr))
        if abbr in s_lower:
            variants.add(s_lower.replace(abbr, full))
    
    # Simple 1-char typos (more systematic)
    if len(s) > 3:
        for i in range(1, len(s)-1):
            # Character deletion
            variants.add(s[:i] + s[i+1:])
            # Character substitution (with common mistakes)
            if s[i].isalpha():
                for char in 'aeiou':  # vowel substitutions are common
                    if char != s[i]:
                        variants.add(s[:i] + char + s[i+1:])
    
    return variants


# Generate variants for all seed vocab
all_seed_variants = {}
for field, seeds in HEADER_VOCAB_VARIATIONS_1.items():
    field_variants = set()
    for s in seeds:
        field_variants.update(gen_basic_variants(s))
    all_seed_variants[field] = sorted(list(field_variants))

save_json(all_seed_variants, "phase1_generated_variants.json", "Programmatic variants for each field")


# -------------------------------
# Step 2: Embedding with metadata
# -------------------------------
# Create mapping for field information
variant_to_field = {}
flat_variants = []
for field, variants in all_seed_variants.items():
    for variant in variants:
        flat_variants.append(variant)
        variant_to_field[variant] = field

# Generate embeddings
print("Generating embeddings...")
flat_emb = model.encode(flat_variants, convert_to_tensor=True, normalize_embeddings=True)
torch.save(flat_emb, os.path.join(OUTPUT_DIR, "phase2_variant_embeddings.pt"))
print(f"Generated embeddings for {len(flat_variants)} variants")


# -------------------------------
# Step 3: Build Enhanced Annoy Index
# -------------------------------
print("Building enhanced Annoy index...")
ann = EnhancedAnnoyIndex(dim=flat_emb.shape[1], metric='angular', n_trees=TREES)

for i, (variant, vec) in enumerate(zip(flat_variants, flat_emb.cpu().numpy())):
    ann.add_item(i, vec, item_data={
        'text': variant,
        'field': variant_to_field[variant],
        'index': i
    })

ann.build()
ann.index.save(os.path.join(OUTPUT_DIR, "phase3_enhanced_annoy_index.ann"))

# Save metadata
metadata = {
    'variant_to_field': variant_to_field,
    'flat_variants': flat_variants,
    'index_config': {
        'trees': TREES,
        'metric': 'angular',
        'dimension': flat_emb.shape[1]
    }
}
save_json(metadata, "phase3_annoy_metadata.json", "Metadata for Annoy index")


# -------------------------------
# Step 4: Advanced Candidate Sampling
# -------------------------------
print("Performing advanced candidate sampling...")
expanded_vocab_advanced = {}

for field, seeds in HEADER_VOCAB_VARIATIONS_1.items():
    print(f"Processing field: {field}")
    field_results = {
        'seeds': seeds,
        'diverse_candidates': set(),
        'random_walk_candidates': set(),
        'radius_candidates': set()
    }
    
    # Generate embeddings for seeds
    seed_embs = model.encode(seeds, convert_to_tensor=True, normalize_embeddings=True)
    
    for i, (seed, emb) in enumerate(zip(seeds, seed_embs)):
        emb_np = emb.cpu().numpy()
        
        # Method 1: Diverse sampling
        diverse_candidates = ann.get_diverse_candidates(
            emb_np, TOP_K, diversity_factor=0.3
        )
        for idx, sim in diverse_candidates:
            if sim >= SIM_THRESHOLD:
                field_results['diverse_candidates'].add(flat_variants[idx])
        
        # Method 2: Random walk sampling
        walk_candidates = ann.sample_random_walk_candidates(emb_np, TOP_K)
        for idx, sim in walk_candidates:
            if sim >= SIM_THRESHOLD:
                field_results['random_walk_candidates'].add(flat_variants[idx])
        
        # Method 3: Radius-based sampling
        radius_candidates = ann.get_all_nns_in_radius(emb_np, SIM_THRESHOLD)
        for idx, sim in radius_candidates:
            field_results['radius_candidates'].add(flat_variants[idx])
    
    # Convert sets to sorted lists
    for key in ['diverse_candidates', 'random_walk_candidates', 'radius_candidates']:
        field_results[key] = sorted(list(field_results[key]))
    
    expanded_vocab_advanced[field] = field_results

save_json(expanded_vocab_advanced, "phase4_advanced_ann_sampling.json", 
         "Advanced ANN sampling results with multiple strategies")


# -------------------------------
# Step 5: Ensemble and Final Processing
# -------------------------------
print("Creating ensemble results...")
def fuzzy_dedupe(phrases, threshold=0.85):
    res = []
    for ph in phrases:
        if all(SequenceMatcher(None, ph.lower(), r.lower()).ratio() < threshold for r in res):
            res.append(ph)
    return res


final_augmented_ensemble = {}
for field, results in expanded_vocab_advanced.items():
    # Combine all sampling methods
    all_candidates = set(results['seeds'])
    all_candidates.update(results['diverse_candidates'])
    all_candidates.update(results['random_walk_candidates'])
    all_candidates.update(results['radius_candidates'])
    
    # Clean and filter
    cleaned = [x.lower().strip() for x in all_candidates]
    cleaned = [p for p in cleaned if 1 < len(p) <= 30 and not p.isdigit()]
    cleaned = list(set(cleaned))  # Remove exact duplicates
    
    # Fuzzy deduplication
    deduplicated = fuzzy_dedupe(cleaned, threshold=0.85)
    
    final_augmented_ensemble[field] = {
        'count': len(deduplicated),
        'variants': sorted(deduplicated)
    }

save_json(final_augmented_ensemble, "phase5_final_ensemble_augmented.json", 
         "Final ensemble results from all sampling strategies")


# -------------------------------
# Analysis and Statistics
# -------------------------------
stats = {
    'original_vocab_size': {field: len(seeds) for field, seeds in HEADER_VOCAB_VARIATIONS_1.items()},
    'augmented_vocab_size': {field: data['count'] for field, data in final_augmented_ensemble.items()},
    'expansion_ratio': {},
    'total_original': sum(len(seeds) for seeds in HEADER_VOCAB_VARIATIONS_1.values()),
    'total_augmented': sum(data['count'] for data in final_augmented_ensemble.values())
}

for field in HEADER_VOCAB_VARIATIONS_1.keys():
    original = stats['original_vocab_size'][field]
    augmented = stats['augmented_vocab_size'][field]
    stats['expansion_ratio'][field] = round(augmented / original, 2) if original > 0 else 0

stats['overall_expansion_ratio'] = round(stats['total_augmented'] / stats['total_original'], 2)

save_json(stats, "phase6_augmentation_statistics.json", "Statistics about vocabulary expansion")

print("\n" + "="*50)
print("PIPELINE COMPLETE")
print("="*50)
print(f"Original vocabulary size: {stats['total_original']}")
print(f"Augmented vocabulary size: {stats['total_augmented']}")
print(f"Overall expansion ratio: {stats['overall_expansion_ratio']}x")
print("\nPer-field expansion ratios:")
for field, ratio in stats['expansion_ratio'].items():
    print(f"  {field}: {ratio}x")
print("="*50)