import hashlib
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CodeSimilarityResult:
    similarity_score: float
    similar_functions: List[Tuple[str, str, float]]  # (func1, func2, score)

class CodeSimilarityAnalyzer:
    def __init__(self, ngram_range: Tuple[int, int] = (3, 5), max_features: int = 10000):
        """Initialize the analyzer with configurable parameters"""
        try:
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb', 
                ngram_range=ngram_range,
                max_features=max_features,
                token_pattern=None  # Use character-based analysis
            )
            self.known_contracts = {}
        except Exception as e:
            print(f"Warning: Failed to initialize vectorizer: {e}")
            # Fallback to basic configuration
            self.vectorizer = TfidfVectorizer()
            self.known_contracts = {}
        
    def _normalize_code(self, code: str) -> str:
        """Normalize code by removing comments and extra whitespace"""
        if not code or not code.strip():
            return ""
        
        try:
            # Remove single-line comments (but not in strings)
            code = re.sub(r'(?<![\"\'])//.*?$', '', code, flags=re.MULTILINE)
            
            # Remove multi-line comments (basic approach - could be improved with proper parsing)
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            
            # Remove pragma and import statements for better similarity detection
            code = re.sub(r'pragma\s+[^;]+;', '', code)
            code = re.sub(r'import\s+[^;]+;', '', code)
            
            # Normalize whitespace
            code = ' '.join(code.split())
            
            return code.lower()  # Case-insensitive comparison
            
        except Exception as e:
            print(f"Warning: Code normalization failed: {e}")
            return code
    
    def _get_function_hashes(self, code: str) -> List[Tuple[str, str]]:
        """Extract function signatures and create hashes"""
        functions = []
        try:
            # More comprehensive regex for function detection
            pattern = r'function\s+([a-zA-Z0-9_]+)\s*\([^{]*\)'
            matches = re.finditer(pattern, code, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                func_signature = match.group(0).strip()
                # Use SHA256 for better hash distribution
                func_hash = hashlib.sha256(func_signature.encode()).hexdigest()[:16]  # Truncate for efficiency
                functions.append((func_signature, func_hash))
                
        except Exception as e:
            # Log error but don't crash
            print(f"Warning: Function extraction failed: {e}")
            
        return functions
    
    def calculate_similarity(self, code1: str, code2: str) -> CodeSimilarityResult:
        """Calculate similarity between two code snippets"""
        if not code1 or not code2:
            return CodeSimilarityResult(similarity_score=0.0, similar_functions=[])
        
        try:
            # Normalize code
            norm1 = self._normalize_code(code1)
            norm2 = self._normalize_code(code2)
            
            if not norm1 or not norm2:
                return CodeSimilarityResult(similarity_score=0.0, similar_functions=[])
            
            # Calculate text similarity with error handling
            try:
                tfidf_matrix = self.vectorizer.fit_transform([norm1, norm2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except Exception as e:
                print(f"Warning: TF-IDF similarity calculation failed: {e}")
                similarity = 0.0
            
            # Get function-level similarities
            funcs1 = self._get_function_hashes(code1)
            funcs2 = self._get_function_hashes(code2)
            
            similar_funcs = []
            for sig1, hash1 in funcs1:
                for sig2, hash2 in funcs2:
                    if hash1 == hash2:
                        similar_funcs.append((sig1, sig2, 1.0))
            
            return CodeSimilarityResult(
                similarity_score=float(similarity),
                similar_functions=similar_funcs
            )
            
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return CodeSimilarityResult(similarity_score=0.0, similar_functions=[])
    
    def compare_with_known_contracts(self, code: str, threshold: float = 0.7) -> Dict[str, float]:
        """Compare code with known contracts in the database"""
        if not code or not self.known_contracts:
            return {}
        
        results = {}
        try:
            for contract_name, contract_code in self.known_contracts.items():
                try:
                    similarity = self.calculate_similarity(code, contract_code).similarity_score
                    if similarity >= threshold:
                        results[contract_name] = round(similarity, 4)  # Round for cleaner output
                except Exception as e:
                    print(f"Warning: Failed to compare with {contract_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in batch comparison: {e}")
            
        return results
    
    def add_known_contract(self, name: str, code: str) -> bool:
        """Add a contract to the known contracts database"""
        if not name or not code:
            return False
        
        try:
            # Validate code can be normalized
            normalized = self._normalize_code(code)
            if normalized:
                self.known_contracts[name] = code
                return True
        except Exception as e:
            print(f"Warning: Failed to add contract {name}: {e}")
        
        return False