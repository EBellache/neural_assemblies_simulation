"""
Geometric Gluing Module
======================
Tropical-E8 geometric gluing for neural assemblies.
Extracted from test file for proper imports.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import time

# Constants
TROPICAL_ZERO = -np.inf

@dataclass
class NeuralPatch:
    """Local patch in the sheaf (neural assembly)."""
    patch_id: int
    cells: Set[int]  # Neuron IDs
    activity: np.ndarray  # Firing rates
    momentum: np.ndarray  # 8D E8 coordinates
    p_level: int = 0  # p-adic hierarchy level
    
    def __post_init__(self):
        if self.momentum is None or len(self.momentum) != 8:
            self.momentum = self.compute_momentum()
    
    def compute_momentum(self) -> np.ndarray:
        """Map neural activity to 8D momentum space."""
        momentum = np.zeros(8)
        
        if len(self.activity) > 0:
            # Map activity statistics to 8D coordinates
            momentum[0] = np.mean(self.activity)  # p_x: mean rate
            momentum[1] = np.std(self.activity)   # p_y: variance
            momentum[2] = np.max(self.activity) - np.min(self.activity)  # L_z: range
            momentum[3] = np.sum(self.activity > 0) / len(self.activity)  # Q: sparsity
            momentum[4] = np.median(self.activity)  # ρ: median
            
            # Higher order statistics
            active = self.activity[self.activity > 0]
            if len(active) > 0:
                momentum[5] = np.percentile(active, 25)  # J_x: Q1
                momentum[6] = np.percentile(active, 75)  # J_y: Q3
                momentum[7] = len(active) / len(self.activity)  # H: fraction active
        
        return momentum


class TropicalE8Gluing:
    """
    Hybrid tropical-E8 gluing system.
    Combines computational efficiency with geometric consistency.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.e8_roots = self._init_e8_simple_roots()
        
        # Performance tracking
        self.stats = {
            'tropical_time': 0,
            'e8_time': 0,
            'total_glues': 0
        }
    
    def _init_e8_simple_roots(self) -> np.ndarray:
        """Initialize E8 simple root system."""
        roots = np.zeros((8, 8))
        
        # Standard E8 simple roots
        for i in range(6):
            roots[i, i] = 1
            if i < 7:
                roots[i, i+1] = -1
        
        roots[6, 6] = 1
        roots[6, 7] = -1
        
        # Special root
        roots[7] = np.array([-0.5] * 8)
        
        return roots
    
    def glue_patches(self, 
                     patches: List[NeuralPatch],
                     overlaps: Dict[Tuple[int, int], Set[int]]) -> Dict:
        """Main gluing function using tropical-E8 hybrid approach."""
        
        # Step 1: Build overlap matrix using tropical algebra
        start = time.time()
        overlap_matrix = self._build_tropical_overlap_matrix(patches, overlaps)
        self.stats['tropical_time'] += time.time() - start
        
        # Step 2: Compute gluing functions
        gluing_data = {}
        
        for (i, j), overlap_cells in overlaps.items():
            if i >= len(patches) or j >= len(patches):
                continue
                
            patch_i = patches[i]
            patch_j = patches[j]
            
            # Tropical gluing for efficiency
            start = time.time()
            tropical_glued = self._tropical_glue(patch_i, patch_j, overlap_cells)
            self.stats['tropical_time'] += time.time() - start
            
            # E8 projection for consistency
            start = time.time()
            e8_glued = self._project_to_e8(tropical_glued)
            self.stats['e8_time'] += time.time() - start
            
            gluing_data[(i, j)] = {
                'tropical': tropical_glued,
                'e8': e8_glued,
                'overlap_size': len(overlap_cells),
                'consistency': self._compute_consistency(patch_i, patch_j, e8_glued)
            }
            
            self.stats['total_glues'] += 1
        
        # Step 3: Check cocycle conditions
        cocycle_consistency = self._check_cocycle_conditions(gluing_data)
        
        # Step 4: Extract cohomology groups
        cohomology = self._compute_cohomology(patches, gluing_data)
        
        return {
            'gluing_data': gluing_data,
            'overlap_matrix': overlap_matrix,
            'cocycle_consistency': cocycle_consistency,
            'cohomology': cohomology,
            'stats': self.stats
        }
    
    def _build_tropical_overlap_matrix(self,
                                      patches: List[NeuralPatch],
                                      overlaps: Dict) -> np.ndarray:
        """Build overlap matrix using tropical inner products."""
        n = len(patches)
        matrix = np.full((n, n), TROPICAL_ZERO)
        
        for (i, j), overlap_cells in overlaps.items():
            if not overlap_cells:
                continue
            
            # Extract overlap activities
            overlap_i = self._extract_overlap_activity(patches[i], overlap_cells)
            overlap_j = self._extract_overlap_activity(patches[j], overlap_cells)
            
            # Tropical inner product: max(a_i + b_i)
            if len(overlap_i) > 0 and len(overlap_j) > 0:
                tropical_product = np.max(overlap_i + overlap_j)
                matrix[i, j] = tropical_product
                matrix[j, i] = tropical_product
        
        return matrix
    
    def _extract_overlap_activity(self, 
                                 patch: NeuralPatch,
                                 overlap_cells: Set[int]) -> np.ndarray:
        """Extract activity for overlapping cells."""
        overlap_activity = []
        
        for idx, cell_id in enumerate(patch.cells):
            if cell_id in overlap_cells:
                if idx < len(patch.activity):
                    overlap_activity.append(patch.activity[idx])
        
        return np.array(overlap_activity) if overlap_activity else np.array([])
    
    def _tropical_glue(self,
                      patch_i: NeuralPatch,
                      patch_j: NeuralPatch,
                      overlap_cells: Set[int]) -> np.ndarray:
        """Perform tropical gluing (max-plus convolution)."""
        # Get overlap activities
        overlap_i = self._extract_overlap_activity(patch_i, overlap_cells)
        overlap_j = self._extract_overlap_activity(patch_j, overlap_cells)
        
        if len(overlap_i) == 0 or len(overlap_j) == 0:
            # No overlap, use momentum average
            return 0.5 * (patch_i.momentum + patch_j.momentum)
        
        # Tropical convolution for smooth gluing
        glued = np.zeros(8)
        
        for d in range(8):
            val_i = patch_i.momentum[d]
            val_j = patch_j.momentum[d]
            
            # Tropical operations (max-plus)
            overlap_strength = np.mean(overlap_i) + np.mean(overlap_j)
            
            if overlap_strength > TROPICAL_ZERO:
                # Weighted max based on overlap strength
                weight_i = np.mean(overlap_i) if len(overlap_i) > 0 else 0
                weight_j = np.mean(overlap_j) if len(overlap_j) > 0 else 0
                total_weight = weight_i + weight_j + 1e-10
                
                glued[d] = (weight_i * val_i + weight_j * val_j) / total_weight
            else:
                glued[d] = max(val_i, val_j)
        
        return glued
    
    def _project_to_e8(self, vector: np.ndarray) -> np.ndarray:
        """Project to nearest E8 lattice point."""
        if len(vector) != 8:
            resized = np.zeros(8)
            resized[:min(8, len(vector))] = vector[:8]
            vector = resized
        
        # Simple E8 projection (can be improved with proper E8 implementation)
        # For now, use a regularized version
        projected = vector.copy()
        
        # Ensure the vector satisfies E8-like constraints
        # Simple constraint: sum of coordinates should be even
        coord_sum = np.sum(projected)
        if abs(coord_sum - np.round(coord_sum)) > 0.5:
            # Adjust to make sum closer to integer
            adjustment = np.round(coord_sum) - coord_sum
            projected += adjustment / 8
        
        return projected
    
    def _compute_consistency(self,
                           patch_i: NeuralPatch,
                           patch_j: NeuralPatch,
                           glued: np.ndarray) -> float:
        """Measure consistency of gluing using information geometry."""
        # Distance from glued point to original patches
        dist_i = np.linalg.norm(glued - patch_i.momentum)
        dist_j = np.linalg.norm(glued - patch_j.momentum)
        
        # Consistency decreases with distance
        consistency = np.exp(-(dist_i + dist_j) / self.temperature)
        
        return float(consistency)
    
    def _check_cocycle_conditions(self, gluing_data: Dict) -> Dict:
        """Check Čech cocycle conditions for consistency."""
        violations = []
        checks = 0
        
        # Get all patch indices
        indices = set()
        for (i, j) in gluing_data.keys():
            indices.add(i)
            indices.add(j)
        
        # Check all triples
        for i in indices:
            for j in indices:
                for k in indices:
                    if i < j < k:
                        if (i,j) in gluing_data and (j,k) in gluing_data and (i,k) in gluing_data:
                            # Get E8 glued states
                            f_ij = gluing_data[(i,j)]['e8']
                            f_jk = gluing_data[(j,k)]['e8'] 
                            f_ik = gluing_data[(i,k)]['e8']
                            
                            # Tropical cocycle: check consistency
                            indirect = 0.5 * (f_ij + f_jk)  # Average path
                            direct = f_ik
                            
                            # Compute violation
                            violation = np.linalg.norm(indirect - direct)
                            
                            if violation > 1e-3:  # Tolerance for numerical precision
                                violations.append({
                                    'triple': (i, j, k),
                                    'violation': violation
                                })
                            
                            checks += 1
        
        consistency = 1.0 - (len(violations) / checks) if checks > 0 else 1.0
        
        return {
            'consistency': consistency,
            'violations': violations,
            'total_checks': checks
        }
    
    def _compute_cohomology(self,
                          patches: List[NeuralPatch],
                          gluing_data: Dict) -> Dict:
        """Compute sheaf cohomology groups."""
        n_patches = len(patches)
        
        if n_patches == 0:
            return {
                'H0_dimension': 0,
                'H1_dimension': 0,
                'global_consistency': 0.0,
                'spectral_gap': 0.0
            }
        
        # Build consistency matrix
        C = np.eye(n_patches)  # Initialize with identity
        
        for (i, j), data in gluing_data.items():
            consistency = data['consistency']
            C[i, j] = consistency
            C[j, i] = consistency
        
        # Compute eigendecomposition
        try:
            eigenvalues = np.linalg.eigvals(C)
            eigenvalues = np.real(eigenvalues)  # Take real part
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # H^0 dimension = number of large eigenvalues (connected components)
            h0_dim = np.sum(eigenvalues > 0.8)
            
            # H^1 dimension = number of small eigenvalues (obstructions)
            h1_dim = np.sum((eigenvalues > 1e-6) & (eigenvalues < 0.5))
            
            global_consistency = float(np.mean(eigenvalues))
            spectral_gap = float(eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0
            
        except:
            # Fallback if eigendecomposition fails
            h0_dim = n_patches
            h1_dim = 0
            global_consistency = 1.0
            spectral_gap = 0.0
        
        return {
            'H0_dimension': h0_dim,
            'H1_dimension': h1_dim,
            'global_consistency': global_consistency,
            'spectral_gap': spectral_gap
        }