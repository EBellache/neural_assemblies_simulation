"""
Piecewise Geodesics in Tropical Geometry
=========================================

Implements piecewise-linear geodesics arising from tropical geometry
for fast approximate computations on morphogenetic manifolds.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Callable
from functools import partial
from dataclasses import dataclass

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class TropicalGeodesic(NamedTuple):
    """Piecewise-linear tropical geodesic."""
    vertices: jnp.ndarray  # Vertices of piecewise path
    edges: List[Tuple[int, int]]  # Edge connectivity
    lengths: jnp.ndarray  # Length of each segment
    total_length: float  # Total tropical length
    is_shortest: bool  # Whether this is a shortest path


class TropicalMetric(NamedTuple):
    """Tropical metric structure."""
    distance_matrix: jnp.ndarray  # Pairwise tropical distances
    dimension: int  # Ambient dimension
    is_ultrametric: bool  # Whether satisfies strong triangle inequality


class PiecewiseFlow(NamedTuple):
    """Piecewise-linear flow on tropical space."""
    trajectory: jnp.ndarray  # Points along flow
    velocities: jnp.ndarray  # Velocity at each point
    switching_times: jnp.ndarray  # Times when flow changes direction
    stability: float  # Stability measure of flow


class TropicalGeodesicComputer:
    """
    Compute piecewise geodesics in tropical geometry.
    """

    def __init__(self, dimension: int = 8):
        """
        Initialize tropical geodesic computer.

        Args:
            dimension: Ambient space dimension
        """
        self.dim = dimension
        self.minus_inf = -1e10

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._tropical_distance = jit(self._tropical_distance_impl)
        self._piecewise_geodesic = jit(self._piecewise_geodesic_impl)
        self._tropical_projection = jit(self._tropical_projection_impl)
        self._gradient_flow = jit(self._gradient_flow_impl)
        self._find_breakpoints = jit(self._find_breakpoints_impl)
        self._tropical_interpolate = jit(self._tropical_interpolate_impl)
        self._compute_tropical_metric = jit(self._compute_tropical_metric_impl)
        self._dijkstra_tropical = jit(self._dijkstra_tropical_impl)

    @partial(jit, static_argnums=(0,))
    def _tropical_distance_impl(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """
        Compute tropical distance.

        d(p, q) = max_i |p_i - q_i|

        Args:
            p, q: Points in tropical space

        Returns:
            Tropical distance
        """
        return jnp.max(jnp.abs(p - q))

    @partial(jit, static_argnums=(0,))
    def _piecewise_geodesic_impl(self, start: jnp.ndarray,
                                 end: jnp.ndarray,
                                 n_segments: int = 10) -> TropicalGeodesic:
        """
        Compute piecewise-linear geodesic.

        Args:
            start: Starting point
            end: Ending point
            n_segments: Number of segments

        Returns:
            Tropical geodesic
        """
        # Find coordinate-wise max differences
        diff = end - start
        max_coord = jnp.argmax(jnp.abs(diff))

        # Tropical geodesic follows max coordinate first
        vertices = [start]
        current = start.copy()

        # Move along maximum coordinate
        intermediate = current.at[max_coord].set(end[max_coord])
        vertices.append(intermediate)

        # Then adjust other coordinates
        vertices.append(end)

        vertices = jnp.array(vertices)
        edges = [(i, i + 1) for i in range(len(vertices) - 1)]

        # Compute segment lengths
        lengths = jnp.array([
            self._tropical_distance_impl(vertices[i], vertices[i + 1])
            for i in range(len(vertices) - 1)
        ])

        total_length = jnp.sum(lengths)

        return TropicalGeodesic(
            vertices=vertices,
            edges=edges,
            lengths=lengths,
            total_length=total_length,
            is_shortest=True
        )

    @partial(jit, static_argnums=(0,))
    def _find_breakpoints_impl(self, function_values: jnp.ndarray,
                               positions: jnp.ndarray) -> jnp.ndarray:
        """
        Find breakpoints where tropical function is non-smooth.

        Args:
            function_values: Values of tropical function
            positions: Positions where evaluated

        Returns:
            Breakpoint locations
        """
        n = len(function_values)
        breakpoints = []

        for i in range(1, n - 1):
            # Check if maximum achieving coordinate changes
            prev_val = function_values[i - 1]
            curr_val = function_values[i]
            next_val = function_values[i + 1]

            # Approximate second derivative
            d2f = next_val - 2 * curr_val + prev_val

            # Large second derivative indicates breakpoint
            if jnp.abs(d2f) > 1e-3:
                breakpoints.append(positions[i])

        return jnp.array(breakpoints) if breakpoints else jnp.array([])

    @partial(jit, static_argnums=(0,))
    def _tropical_projection_impl(self, point: jnp.ndarray,
                                  tropical_variety: jnp.ndarray) -> jnp.ndarray:
        """
        Project point onto tropical variety.

        Args:
            point: Point to project
            tropical_variety: Points defining variety

        Returns:
            Projected point
        """
        # Find closest point in tropical metric
        distances = vmap(lambda p: self._tropical_distance_impl(point, p))(tropical_variety)
        closest_idx = jnp.argmin(distances)

        closest = tropical_variety[closest_idx]

        # Refine by moving along tropical line
        direction = point - closest
        max_coord = jnp.argmax(jnp.abs(direction))

        # Project along maximum coordinate
        projected = closest.copy()
        projected = projected.at[max_coord].set(point[max_coord])

        return projected

    @partial(jit, static_argnums=(0,))
    def _gradient_flow_impl(self, initial: jnp.ndarray,
                            potential: Callable,
                            dt: float = 0.01,
                            n_steps: int = 100) -> PiecewiseFlow:
        """
        Compute tropical gradient flow.

        Args:
            initial: Initial point
            potential: Potential function
            dt: Time step
            n_steps: Number of steps

        Returns:
            Piecewise flow
        """
        trajectory = [initial]
        velocities = []
        switching_times = []

        current = initial
        prev_velocity = jnp.zeros_like(initial)

        for step in range(n_steps):
            # Tropical gradient (subdifferential)
            grad_val = grad(potential)(current)

            # Find maximum component (tropical choice)
            max_idx = jnp.argmax(jnp.abs(grad_val))

            # Velocity in tropical sense (move along max gradient)
            velocity = jnp.zeros_like(current)
            velocity = velocity.at[max_idx].set(-grad_val[max_idx])

            # Detect switching
            if not jnp.allclose(velocity, prev_velocity, rtol=1e-4):
                switching_times.append(step * dt)

            # Update position
            current = current + dt * velocity

            trajectory.append(current)
            velocities.append(velocity)
            prev_velocity = velocity

        trajectory = jnp.array(trajectory)
        velocities = jnp.array(velocities)
        switching_times = jnp.array(switching_times) if switching_times else jnp.array([])

        # Compute stability (Lyapunov-like)
        stability = -jnp.mean(jnp.array([
            potential(trajectory[i + 1]) - potential(trajectory[i])
            for i in range(len(trajectory) - 1)
        ]))

        return PiecewiseFlow(
            trajectory=trajectory,
            velocities=velocities,
            switching_times=switching_times,
            stability=stability
        )

    @partial(jit, static_argnums=(0,))
    def _tropical_interpolate_impl(self, path: jnp.ndarray,
                                   t: float) -> jnp.ndarray:
        """
        Interpolate along tropical path.

        Args:
            path: Piecewise linear path
            t: Parameter in [0, 1]

        Returns:
            Interpolated point
        """
        n_segments = len(path) - 1
        if n_segments == 0:
            return path[0]

        # Find which segment t falls in
        segment_idx = jnp.clip(int(t * n_segments), 0, n_segments - 1)
        local_t = t * n_segments - segment_idx

        # Linear interpolation within segment
        start = path[segment_idx]
        end = path[segment_idx + 1]

        # Tropical interpolation: move along max coordinate first
        diff = end - start
        max_coord = jnp.argmax(jnp.abs(diff))

        interpolated = start.copy()

        if local_t < 0.5:
            # First half: move along max coordinate
            interpolated = interpolated.at[max_coord].set(
                start[max_coord] + 2 * local_t * diff[max_coord]
            )
        else:
            # Second half: adjust other coordinates
            interpolated = interpolated.at[max_coord].set(end[max_coord])
            for i in range(self.dim):
                if i != max_coord:
                    interpolated = interpolated.at[i].set(
                        start[i] + 2 * (local_t - 0.5) * diff[i]
                    )

        return interpolated

    @partial(jit, static_argnums=(0,))
    def _compute_tropical_metric_impl(self, points: jnp.ndarray) -> TropicalMetric:
        """
        Compute tropical metric structure from points.

        Args:
            points: Collection of points

        Returns:
            Tropical metric
        """
        n = len(points)
        distance_matrix = jnp.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist = self._tropical_distance_impl(points[i], points[j])
                distance_matrix = distance_matrix.at[i, j].set(dist)

        # Check ultrametric property
        is_ultrametric = True
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    dij = distance_matrix[i, j]
                    djk = distance_matrix[j, k]
                    dik = distance_matrix[i, k]

                    # Strong triangle inequality: d(i,k) ≤ max(d(i,j), d(j,k))
                    if dik > jnp.maximum(dij, djk) + 1e-6:
                        is_ultrametric = False
                        break

        return TropicalMetric(
            distance_matrix=distance_matrix,
            dimension=self.dim,
            is_ultrametric=is_ultrametric
        )

    @partial(jit, static_argnums=(0,))
    def _dijkstra_tropical_impl(self, adjacency: jnp.ndarray,
                                source: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Dijkstra's algorithm with tropical metric.

        Args:
            adjacency: Weighted adjacency matrix (tropical weights)
            source: Source vertex

        Returns:
            (distances, predecessors)
        """
        n = adjacency.shape[0]
        distances = jnp.full(n, jnp.inf)
        distances = distances.at[source].set(0)
        predecessors = jnp.full(n, -1)
        visited = jnp.zeros(n, dtype=bool)

        for _ in range(n):
            # Find unvisited vertex with minimum distance
            unvisited_distances = jnp.where(visited, jnp.inf, distances)
            u = jnp.argmin(unvisited_distances)

            if distances[u] == jnp.inf:
                break

            visited = visited.at[u].set(True)

            # Update distances to neighbors
            for v in range(n):
                if not visited[v] and adjacency[u, v] < jnp.inf:
                    # Tropical distance: max operation
                    alt = jnp.maximum(distances[u], adjacency[u, v])

                    if alt < distances[v]:
                        distances = distances.at[v].set(alt)
                        predecessors = predecessors.at[v].set(u)

        return distances, predecessors

    def compute_tropical_voronoi(self, sites: jnp.ndarray,
                                 domain_points: jnp.ndarray) -> Dict[int, jnp.ndarray]:
        """
        Compute tropical Voronoi diagram.

        Args:
            sites: Voronoi sites
            domain_points: Points to classify

        Returns:
            Dictionary mapping site index to region points
        """
        voronoi = {i: [] for i in range(len(sites))}

        for point in domain_points:
            distances = vmap(lambda s: self._tropical_distance(point, s))(sites)
            nearest = jnp.argmin(distances)
            voronoi[int(nearest)].append(point)

        # Convert lists to arrays
        for i in voronoi:
            if voronoi[i]:
                voronoi[i] = jnp.array(voronoi[i])
            else:
                voronoi[i] = jnp.array([]).reshape(0, self.dim)

        return voronoi