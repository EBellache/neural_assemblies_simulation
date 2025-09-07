"""Tropical polytope and amoeba structures.

The tropical polytope with 7±2 vertices forms the geometric foundation
for assembly organization. The amoeba thickness τ ≈ 2 explains the 
natural variation in assembly count.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from functools import partial


class PolytopeVertex(NamedTuple):
    """Single vertex of the tropical polytope."""
    index: int
    position: jnp.ndarray  # Position in ambient space
    eigenvalue: float  # Assembly specialization parameter
    metabolic_cost: float  # Energy required to maintain


class TropicalPolytope:
    """Tropical polytope with 7±2 vertices for assembly organization.
    
    The vertices correspond to stable assembly states, edges to transitions,
    and faces to multi-assembly competitions.
    """
    
    def __init__(self, 
                 n_vertices: int = 7,
                 dimension: int = 6,
                 amoeba_thickness: float = 2.0):
        """Initialize tropical polytope.
        
        Args:
            n_vertices: Number of vertices (5-9, default 7)
            dimension: Ambient dimension (default 6 for 6-simplex)
            amoeba_thickness: Thickness parameter τ
        """
        assert 5 <= n_vertices <= 9, "Vertices must be in range [5, 9]"
        
        self.n_vertices = n_vertices
        self.dimension = dimension
        self.amoeba_thickness = amoeba_thickness
        
        # Initialize vertices
        self.vertices = self._initialize_vertices()
        
        # Compute edge matrix (transition costs)
        self.edge_matrix = self._compute_edge_matrix()
        
    def _initialize_vertices(self) -> List[PolytopeVertex]:
        """Initialize polytope vertices.
        
        For 7 vertices, use regular 6-simplex.
        For other counts, perturb or add/remove vertices.
        """
        vertices = []
        
        if self.n_vertices == 7:
            # Regular 6-simplex vertices
            positions = self._regular_simplex_vertices(self.dimension)
            eigenvalues = jnp.linspace(2.0, 0.9, 7)
        else:
            # Adjust vertex count
            base_positions = self._regular_simplex_vertices(6)
            if self.n_vertices < 7:
                # Remove vertices
                positions = base_positions[:self.n_vertices]
            else:
                # Add vertices by subdivision
                positions = self._subdivide_vertices(base_positions, self.n_vertices)
            eigenvalues = jnp.linspace(2.0, 0.9, self.n_vertices)
            
        # Create vertex objects
        for i in range(self.n_vertices):
            vertex = PolytopeVertex(
                index=i,
                position=positions[i],
                eigenvalue=eigenvalues[i],
                metabolic_cost=self._compute_metabolic_cost(eigenvalues[i])
            )
            vertices.append(vertex)
            
        return vertices
        
    @staticmethod
    def _regular_simplex_vertices(dim: int) -> jnp.ndarray:
        """Generate vertices of regular simplex in dim dimensions.
        
        Returns:
            Array of shape (dim+1, dim) with vertex coordinates
        """
        n = dim + 1
        vertices = jnp.zeros((n, dim))
        
        # Standard simplex construction
        for i in range(n):
            if i < dim:
                vertices = vertices.at[i, i].set(1.0)
            else:
                # Last vertex at center with offset
                vertices = vertices.at[i].set(-1.0 / jnp.sqrt(dim))
                
        # Normalize to unit simplex
        center = jnp.mean(vertices, axis=0)
        vertices = vertices - center
        
        return vertices
        
    @staticmethod
    def _subdivide_vertices(base_vertices: jnp.ndarray, 
                           target_count: int) -> jnp.ndarray:
        """Add vertices by edge subdivision."""
        vertices = list(base_vertices)
        
        while len(vertices) < target_count:
            # Add midpoint of longest edge
            max_dist = 0
            max_pair = (0, 1)
            
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    dist = jnp.linalg.norm(vertices[i] - vertices[j])
                    if dist > max_dist:
                        max_dist = dist
                        max_pair = (i, j)
                        
            # Add midpoint
            midpoint = (vertices[max_pair[0]] + vertices[max_pair[1]]) / 2
            vertices.append(midpoint)
            
        return jnp.array(vertices[:target_count])
        
    @staticmethod  
    def _compute_metabolic_cost(eigenvalue: float) -> float:
        """Compute metabolic cost based on eigenvalue.
        
        Higher eigenvalues (elongated structures) have lower base cost
        but higher activation cost.
        """
        base_cost = 2.0 - eigenvalue  # Lower for specialized detectors
        activation_cost = eigenvalue * 0.5  # Higher for complex features
        return base_cost + activation_cost
        
    @jax.jit
    def _compute_edge_matrix(self) -> jnp.ndarray:
        """Compute edge weights (transition costs) between vertices."""
        n = self.n_vertices
        edge_matrix = jnp.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Tropical distance between vertices
                    dist = jnp.max(jnp.abs(
                        self.vertices[i].position - self.vertices[j].position
                    ))
                    
                    # Metabolic transition cost
                    metabolic_diff = jnp.abs(
                        self.vertices[i].metabolic_cost - 
                        self.vertices[j].metabolic_cost
                    )
                    
                    edge_matrix = edge_matrix.at[i, j].set(dist + metabolic_diff)
                    
        return edge_matrix
        
    @partial(jax.jit, static_argnames=['self'])
    def vertex_competition(self, input_scores: jnp.ndarray) -> Tuple[int, float]:
        """Tropical competition between vertices.
        
        Args:
            input_scores: Score for each vertex
            
        Returns:
            (winner_index, confidence)
        """
        # Apply metabolic modulation
        effective_scores = input_scores - jnp.array([
            v.metabolic_cost for v in self.vertices
        ])
        
        # Tropical maximum (winner-take-all)
        winner = jnp.argmax(effective_scores)
        
        # Confidence from margin
        sorted_scores = jnp.sort(effective_scores)
        confidence = sorted_scores[-1] - sorted_scores[-2]
        
        return winner, confidence
        
    def adjust_vertices(self, metabolic_state: float) -> int:
        """Adjust vertex count based on metabolic state.
        
        Args:
            metabolic_state: Current metabolic level (0-1)
            
        Returns:
            Adjusted number of active vertices
        """
        # Use amoeba thickness to determine vertex count
        base_vertices = 7
        variation = jnp.floor(self.amoeba_thickness * (metabolic_state - 0.5))
        n_vertices = jnp.clip(base_vertices + variation, 5, 9).astype(int)
        
        return n_vertices


class Amoeba:
    """Amoeba of a tropical variety with finite thickness.
    
    The amoeba is the image of an algebraic variety under the logarithmic map.
    Its thickness τ ≈ 2 explains the 7±2 assembly variation.
    """
    
    def __init__(self, 
                 polytope: TropicalPolytope,
                 thickness: float = 2.0):
        """Initialize amoeba from tropical polytope.
        
        Args:
            polytope: Underlying tropical polytope
            thickness: Amoeba thickness τ
        """
        self.polytope = polytope
        self.thickness = thickness
        
        # Compute spine (tropical variety skeleton)
        self.spine = self._compute_spine()
        
        # Tentacle directions (extending to infinity)
        self.tentacles = self._compute_tentacles()
        
    def _compute_spine(self) -> jnp.ndarray:
        """Compute the spine (skeleton) of the amoeba.
        
        The spine consists of the edges of the tropical variety.
        """
        edges = []
        
        for i in range(self.polytope.n_vertices):
            for j in range(i+1, self.polytope.n_vertices):
                # Edge from vertex i to vertex j
                edge = jnp.stack([
                    self.polytope.vertices[i].position,
                    self.polytope.vertices[j].position
                ])
                edges.append(edge)
                
        return jnp.array(edges)
        
    def _compute_tentacles(self) -> jnp.ndarray:
        """Compute tentacle directions.
        
        Tentacles extend from vertices toward infinity.
        """
        tentacles = []
        
        for vertex in self.polytope.vertices:
            # Tentacle direction based on eigenvalue
            direction = jnp.zeros(self.polytope.dimension)
            direction = direction.at[0].set(vertex.eigenvalue)
            direction = direction / jnp.linalg.norm(direction)
            tentacles.append(direction)
            
        return jnp.array(tentacles)
        
    @jax.jit
    def contains_point(self, point: jnp.ndarray) -> bool:
        """Check if point is within amoeba (with thickness).
        
        Args:
            point: Point to test
            
        Returns:
            True if point is within amoeba body
        """
        # Distance to nearest spine edge
        min_dist = jnp.inf
        
        for edge in self.spine:
            # Distance from point to line segment
            v = edge[1] - edge[0]
            w = point - edge[0]
            
            t = jnp.clip(jnp.dot(w, v) / jnp.dot(v, v), 0, 1)
            projection = edge[0] + t * v
            dist = jnp.linalg.norm(point - projection)
            
            min_dist = jnp.minimum(min_dist, dist)
            
        return min_dist <= self.thickness / 2
        
    @jax.jit
    def modulate_thickness(self, 
                          atp_level: float,
                          temperature: float = 37.0) -> float:
        """Modulate amoeba thickness based on metabolic state.
        
        Args:
            atp_level: ATP concentration (normalized 0-1)
            temperature: Temperature in Celsius
            
        Returns:
            Modulated thickness
        """
        # Temperature factor (optimal at 37°C)
        temp_factor = jnp.exp(-jnp.abs(temperature - 37.0) / 5.0)
        
        # ATP factor (increases thickness with energy)
        atp_factor = jnp.sqrt(atp_level)
        
        # Modulated thickness
        tau = self.thickness * temp_factor * atp_factor
        
        return tau
        
    def assembly_capacity(self, tau: float) -> int:
        """Determine assembly capacity from thickness.
        
        Args:
            tau: Current amoeba thickness
            
        Returns:
            Number of assemblies (5-9)
        """
        # Base 7 with ±floor(tau) variation
        variation = jnp.floor(tau)
        capacity = jnp.clip(7 + jax.random.randint(
            jax.random.PRNGKey(0), (), -variation, variation+1
        ), 5, 9)
        
        return int(capacity)