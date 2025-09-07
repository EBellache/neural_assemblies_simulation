"""Tests for tropical operations."""

import pytest
import numpy as np
import jax.numpy as jnp

from tropical_sdr.core.tropical_ops import (
    tropical_add,
    tropical_mul,
    tropical_dot,
    tropical_distance,
    tropical_polynomial,
    tropical_matrix_mult,
    TROPICAL_NEG_INF
)


class TestTropicalOperations:
    """Test tropical arithmetic operations."""
    
    def test_tropical_add(self):
        """Test tropical addition (max operation)."""
        # Scalar tests
        assert tropical_add(3.0, 5.0) == 5.0
        assert tropical_add(-1.0, 2.0) == 2.0
        assert tropical_add(TROPICAL_NEG_INF, 5.0) == 5.0
        
        # Array tests
        x = jnp.array([1, 2, 3])
        y = jnp.array([3, 2, 1])
        result = tropical_add(x, y)
        expected = jnp.array([3, 2, 3])
        assert jnp.allclose(result, expected)
        
    def test_tropical_mul(self):
        """Test tropical multiplication (addition)."""
        # Scalar tests
        assert tropical_mul(3.0, 5.0) == 8.0
        assert tropical_mul(-1.0, 2.0) == 1.0
        
        # Test with tropical -inf
        assert tropical_mul(TROPICAL_NEG_INF, 5.0) == TROPICAL_NEG_INF
        
        # Array tests
        x = jnp.array([1, 2, 3])
        y = jnp.array([3, 2, 1])
        result = tropical_mul(x, y)
        expected = jnp.array([4, 4, 4])
        assert jnp.allclose(result, expected)
        
    def test_tropical_dot(self):
        """Test tropical inner product."""
        x = jnp.array([1, 2, 3])
        y = jnp.array([3, 2, 1])
        
        # Should compute max(1+3, 2+2, 3+1) = max(4, 4, 4) = 4
        result = tropical_dot(x, y)
        assert result == 4.0
        
        # Test with different values
        x = jnp.array([1, 5, 2])
        y = jnp.array([2, 1, 3])
        # max(1+2, 5+1, 2+3) = max(3, 6, 5) = 6
        result = tropical_dot(x, y)
        assert result == 6.0
        
    def test_tropical_distance(self):
        """Test tropical distance."""
        x = jnp.array([1, 2, 3])
        y = jnp.array([3, 2, 1])
        
        # max(|1-3|, |2-2|, |3-1|) = max(2, 0, 2) = 2
        result = tropical_distance(x, y)
        assert result == 2.0
        
    def test_tropical_polynomial(self):
        """Test tropical polynomial evaluation."""
        # P(x) = max(2 + 1*x, 3 + 2*x)
        coeffs = jnp.array([2, 3])
        powers = jnp.array([1, 2])
        
        x = jnp.array([1.0])
        # max(2 + 1*1, 3 + 2*1) = max(3, 5) = 5
        result = tropical_polynomial(coeffs, powers, x)
        assert result == 5.0
        
    def test_tropical_matrix_mult(self):
        """Test tropical matrix multiplication."""
        A = jnp.array([[1, 2], [3, 4]])
        B = jnp.array([[2, 1], [1, 2]])
        
        # Result[i,j] = max_k(A[i,k] + B[k,j])
        # Result[0,0] = max(1+2, 2+1) = max(3, 3) = 3
        # Result[0,1] = max(1+1, 2+2) = max(2, 4) = 4
        # Result[1,0] = max(3+2, 4+1) = max(5, 5) = 5
        # Result[1,1] = max(3+1, 4+2) = max(4, 6) = 6
        
        result = tropical_matrix_mult(A, B)
        expected = jnp.array([[3, 4], [5, 6]])
        assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])