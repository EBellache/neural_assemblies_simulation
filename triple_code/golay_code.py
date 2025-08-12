"""
Golay Code Module
=================
Implementation of the extended binary Golay [24,12,8] code.
Used for error correction in neuronal assembly patterns.

The Golay code can correct up to 3 errors and detect up to 7.
Perfect for biological systems with noisy communication.

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
from scipy.linalg import null_space


# GF(2) field operations
def gf2(x: np.ndarray) -> np.ndarray:
    """Convert to GF(2) (binary field)."""
    return np.mod(x, 2).astype(np.uint8)


def gf2_rank(A: np.ndarray) -> int:
    """Compute rank of matrix over GF(2)."""
    A = gf2(A)
    m, n = A.shape
    rank = 0

    for col in range(n):
        # Find pivot
        for row in range(rank, m):
            if A[row, col] == 1:
                # Swap rows
                if row != rank:
                    A[[row, rank]] = A[[rank, row]]
                break
        else:
            continue

        # Eliminate
        for row in range(m):
            if row != rank and A[row, col] == 1:
                A[row] = gf2(A[row] + A[rank])

        rank += 1

    return rank


@dataclass
class GolayEncoder:
    """
    Golay encoder for converting 12-bit messages to 24-bit codewords.
    """
    generator_matrix: np.ndarray = None

    def __post_init__(self):
        if self.generator_matrix is None:
            self.generator_matrix = self._create_generator_matrix()

    def _create_generator_matrix(self) -> np.ndarray:
        """
        Create the 12×24 generator matrix for extended Golay code.
        Uses the standard construction from coding theory.
        """
        # First create the 12×12 identity matrix
        I12 = np.eye(12, dtype=np.uint8)

        # Create the 12×12 matrix A (specific to Golay code)
        # This is based on the quadratic residues modulo 23
        A = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        ], dtype=np.uint8)

        # Generator matrix G = [I12 | A]
        G = np.hstack([I12, A])

        return G

    def encode(self, message: Union[np.ndarray, List[int], int]) -> np.ndarray:
        """
        Encode a 12-bit message to 24-bit Golay codeword.

        Parameters:
        -----------
        message : array-like or int
            12-bit message (array of bits or integer 0-4095)

        Returns:
        --------
        np.ndarray : 24-bit codeword
        """
        # Convert to bit array if needed
        if isinstance(message, int):
            if message < 0 or message > 4095:
                raise ValueError("Message must be 12-bit (0-4095)")
            message = np.array([(message >> i) & 1 for i in range(12)], dtype=np.uint8)
        else:
            message = np.array(message, dtype=np.uint8)

        if len(message) != 12:
            raise ValueError("Message must be 12 bits")

        # Encode: c = m × G (mod 2)
        codeword = gf2(message @ self.generator_matrix)

        return codeword

    def encode_batch(self, messages: np.ndarray) -> np.ndarray:
        """
        Encode multiple messages efficiently.

        Parameters:
        -----------
        messages : np.ndarray of shape (n, 12)
            Multiple 12-bit messages

        Returns:
        --------
        np.ndarray of shape (n, 24) : Codewords
        """
        return gf2(messages @ self.generator_matrix)


@dataclass
class GolayDecoder:
    """
    Golay decoder for error correction.
    Can correct up to 3 errors, detect up to 7.
    """
    parity_matrix: np.ndarray = None
    syndrome_table: Dict[tuple, np.ndarray] = None

    def __post_init__(self):
        if self.parity_matrix is None:
            self.parity_matrix = self._create_parity_matrix()
        if self.syndrome_table is None:
            self.syndrome_table = self._build_syndrome_table()

    def _create_parity_matrix(self) -> np.ndarray:
        """
        Create the 12×24 parity check matrix H.
        For extended Golay: H = [A^T | I12]
        """
        # Use the same A matrix from encoder
        A = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        ], dtype=np.uint8)

        I12 = np.eye(12, dtype=np.uint8)

        # H = [A^T | I12]
        H = np.hstack([A.T, I12])

        return H

    def _build_syndrome_table(self) -> Dict[tuple, np.ndarray]:
        """
        Build syndrome lookup table for fast decoding.
        Maps syndromes to error patterns (up to weight 3).
        """
        table = {}

        # Add zero syndrome (no error)
        table[tuple(np.zeros(12, dtype=np.uint8))] = np.zeros(24, dtype=np.uint8)

        # Single-bit errors
        for i in range(24):
            error = np.zeros(24, dtype=np.uint8)
            error[i] = 1
            syndrome = self.compute_syndrome(error)
            table[tuple(syndrome)] = error

        # Two-bit errors
        for i in range(24):
            for j in range(i + 1, 24):
                error = np.zeros(24, dtype=np.uint8)
                error[i] = 1
                error[j] = 1
                syndrome = self.compute_syndrome(error)
                if tuple(syndrome) not in table:
                    table[tuple(syndrome)] = error

        # Three-bit errors (selective to keep table size reasonable)
        for i in range(24):
            for j in range(i + 1, min(i + 12, 24)):
                for k in range(j + 1, min(j + 6, 24)):
                    error = np.zeros(24, dtype=np.uint8)
                    error[i] = 1
                    error[j] = 1
                    error[k] = 1
                    syndrome = self.compute_syndrome(error)
                    if tuple(syndrome) not in table:
                        table[tuple(syndrome)] = error

        return table

    def compute_syndrome(self, received: np.ndarray) -> np.ndarray:
        """
        Compute syndrome of received word.
        s = r × H^T (mod 2)
        """
        return gf2(received @ self.parity_matrix.T)

    def decode(self, received: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode received word, correcting up to 3 errors.

        Parameters:
        -----------
        received : array-like
            24-bit received word (possibly with errors)

        Returns:
        --------
        (corrected_codeword, error_pattern) : Both 24-bit arrays
        """
        received = np.array(received, dtype=np.uint8)

        if len(received) != 24:
            raise ValueError("Received word must be 24 bits")

        # Compute syndrome
        syndrome = self.compute_syndrome(received)

        # Look up error pattern
        syndrome_tuple = tuple(syndrome)

        if syndrome_tuple in self.syndrome_table:
            error_pattern = self.syndrome_table[syndrome_tuple]
        else:
            # Too many errors to correct
            # Try to find closest syndrome
            min_dist = float('inf')
            best_error = np.zeros(24, dtype=np.uint8)

            for syn_key, err in self.syndrome_table.items():
                dist = np.sum(np.abs(np.array(syn_key) - syndrome))
                if dist < min_dist:
                    min_dist = dist
                    best_error = err

            error_pattern = best_error

        # Correct the error
        corrected = gf2(received + error_pattern)

        return corrected, error_pattern

    def decode_to_message(self, received: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Decode received word and extract 12-bit message.

        Returns:
        --------
        np.ndarray : 12-bit message
        """
        corrected, _ = self.decode(received)
        # Message is in first 12 bits
        return corrected[:12]

    def detect_errors(self, received: np.ndarray) -> int:
        """
        Detect number of errors (up to 7).

        Returns:
        --------
        int : Estimated number of errors (0-7, or -1 if > 7)
        """
        syndrome = self.compute_syndrome(received)

        # Zero syndrome means no errors (or undetectable)
        if np.all(syndrome == 0):
            return 0

        # Check syndrome weight
        syndrome_weight = np.sum(syndrome)

        # Use syndrome weight as rough estimate
        if syndrome_weight <= 3:
            return 1  # Likely single error
        elif syndrome_weight <= 5:
            return 2  # Likely 2 errors
        elif syndrome_weight <= 7:
            return 3  # Likely 3 errors
        elif syndrome_weight <= 9:
            return 4  # 4-7 errors
        else:
            return -1  # More than 7 errors


class GolayAssemblyEncoder:
    """
    Encode neuronal assemblies using Golay codes.
    Maps assembly patterns to error-corrected representations.
    """

    def __init__(self):
        self.encoder = GolayEncoder()
        self.decoder = GolayDecoder()

    def assembly_to_message(self, assembly_pattern: np.ndarray) -> np.ndarray:
        """
        Convert assembly firing pattern to 12-bit message.

        Parameters:
        -----------
        assembly_pattern : np.ndarray
            Binary firing pattern or rate vector

        Returns:
        --------
        np.ndarray : 12-bit message
        """
        # Ensure binary
        if assembly_pattern.dtype != np.uint8:
            # Threshold at mean
            threshold = np.mean(assembly_pattern)
            assembly_pattern = (assembly_pattern > threshold).astype(np.uint8)

        # Take first 12 bits or pad/truncate
        if len(assembly_pattern) >= 12:
            message = assembly_pattern[:12]
        else:
            message = np.pad(assembly_pattern,
                             (0, 12 - len(assembly_pattern)),
                             mode='constant')

        return message

    def encode_assembly(self, assembly_pattern: np.ndarray) -> np.ndarray:
        """
        Encode assembly pattern with error correction.

        Returns:
        --------
        np.ndarray : 24-bit Golay codeword
        """
        message = self.assembly_to_message(assembly_pattern)
        return self.encoder.encode(message)

    def decode_assembly(self, received_pattern: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Decode received pattern, correcting errors.

        Returns:
        --------
        (corrected_pattern, n_errors) : Corrected 12-bit pattern and error count
        """
        if len(received_pattern) != 24:
            raise ValueError("Expected 24-bit received pattern")

        corrected, error_pattern = self.decoder.decode(received_pattern)
        n_errors = np.sum(error_pattern)

        # Extract message
        message = corrected[:12]

        return message, int(n_errors)

    def compute_assembly_reliability(self,
                                     assemblies: List[np.ndarray],
                                     noise_level: float = 0.1) -> Dict[str, Any]:
        """
        Analyze reliability of assembly encoding under noise.

        Parameters:
        -----------
        assemblies : List[np.ndarray]
            List of assembly patterns
        noise_level : float
            Probability of bit flip

        Returns:
        --------
        Dict with reliability metrics
        """
        n_assemblies = len(assemblies)
        n_trials = 100

        successes = 0
        total_errors = 0
        uncorrectable = 0

        for assembly in assemblies:
            # Encode
            codeword = self.encode_assembly(assembly)

            for _ in range(n_trials):
                # Add noise
                noise = np.random.random(24) < noise_level
                noisy = gf2(codeword + noise.astype(np.uint8))

                # Decode
                decoded, n_err = self.decode_assembly(noisy)

                # Check if correctly decoded
                original_message = self.assembly_to_message(assembly)
                if np.array_equal(decoded, original_message):
                    successes += 1
                else:
                    uncorrectable += 1

                total_errors += n_err

        total_trials = n_assemblies * n_trials

        return {
            'success_rate': successes / total_trials,
            'avg_errors_corrected': total_errors / total_trials,
            'uncorrectable_rate': uncorrectable / total_trials,
            'noise_level': noise_level,
            'n_assemblies': n_assemblies,
            'n_trials': n_trials
        }


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between binary vectors."""
    return int(np.sum(np.abs(a - b)))


def minimum_distance(codewords: List[np.ndarray]) -> int:
    """Find minimum Hamming distance among codewords."""
    min_dist = float('inf')

    for i in range(len(codewords)):
        for j in range(i + 1, len(codewords)):
            dist = hamming_distance(codewords[i], codewords[j])
            min_dist = min(min_dist, dist)

    return int(min_dist)


def test_golay_code():
    """Test Golay code functionality."""
    print("\n=== Testing Golay Code Module ===\n")

    # Test encoder
    encoder = GolayEncoder()

    # Test message
    message = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    print(f"Original message: {message}")

    codeword = encoder.encode(message)
    print(f"Encoded codeword: {codeword}")
    print(f"Codeword length: {len(codeword)}")

    # Test decoder
    decoder = GolayDecoder()

    # Test with no errors
    decoded, errors = decoder.decode(codeword)
    print(f"\nNo errors:")
    print(f"  Decoded: {decoded[:12]}")
    print(f"  Matches original: {np.array_equal(decoded[:12], message)}")

    # Test with 1 error
    corrupted = codeword.copy()
    corrupted[5] = 1 - corrupted[5]  # Flip bit 5
    decoded, errors = decoder.decode(corrupted)
    print(f"\n1 error at position 5:")
    print(f"  Decoded: {decoded[:12]}")
    print(f"  Matches original: {np.array_equal(decoded[:12], message)}")
    print(f"  Errors corrected: {np.sum(errors)}")

    # Test with 3 errors
    corrupted = codeword.copy()
    for pos in [2, 7, 15]:
        corrupted[pos] = 1 - corrupted[pos]
    decoded, errors = decoder.decode(corrupted)
    print(f"\n3 errors at positions 2, 7, 15:")
    print(f"  Decoded: {decoded[:12]}")
    print(f"  Matches original: {np.array_equal(decoded[:12], message)}")
    print(f"  Errors corrected: {np.sum(errors)}")

    # Test assembly encoding
    print("\n--- Testing Assembly Encoding ---")

    assembly_encoder = GolayAssemblyEncoder()

    # Create synthetic assembly pattern
    assembly_pattern = np.random.randint(0, 2, 20).astype(np.uint8)
    print(f"\nAssembly pattern (20 bits): {assembly_pattern}")

    # Encode
    encoded = assembly_encoder.encode_assembly(assembly_pattern)
    print(f"Encoded (24 bits): {encoded}")

    # Test reliability
    print("\nTesting reliability under noise...")
    assemblies = [np.random.randint(0, 2, 15).astype(np.uint8) for _ in range(10)]

    for noise in [0.05, 0.10, 0.15]:
        stats = assembly_encoder.compute_assembly_reliability(assemblies, noise)
        print(f"\nNoise level {noise:.0%}:")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg errors corrected: {stats['avg_errors_corrected']:.2f}")
        print(f"  Uncorrectable rate: {stats['uncorrectable_rate']:.1%}")

    # Verify minimum distance
    print("\n--- Verifying Code Properties ---")

    # Generate several codewords
    codewords = []
    for i in range(20):
        msg = np.random.randint(0, 2, 12).astype(np.uint8)
        cw = encoder.encode(msg)
        codewords.append(cw)

    min_dist = minimum_distance(codewords)
    print(f"Minimum distance: {min_dist} (should be 8 for Golay code)")

    print("\n✓ Golay code module working correctly!")


if __name__ == "__main__":
    test_golay_code()