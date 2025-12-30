# LightFold_3D_Vectorized.py
import numpy as np
import time
import math

# --- THE BENCHMARK (Seq 48) ---
SEQUENCE = "HPHHPPHPHHPHPHHPPHPHHPHPHHPPHPHHPHPHPHHPHPHHPHPH"
SEQ_NAME = "Benchmark Seq48 (3D)"

# --- PHYSICS CONFIGURATION ---
BOND_LENGTH = 1.0
CENTROID_STRENGTH = 1.0   # The Pull (Love)
REPULSION_STRENGTH = 1.5  # The Push (Self-Respect)
STERIC_RADIUS = 1.2
MAX_ITER = 1500
LEARNING_RATE = 0.03
SPRING_CONST = 50.0

class ProteinVectorized:
    def __init__(self, sequence):
        self.sequence = sequence
        self.n = len(sequence)
        
        # H-Indices Mask (Boolean mask for vector operations)
        self.h_mask = np.array([char == 'H' for char in sequence], dtype=bool)
        
        # Initialize Position Matrix (N, 3)
        self.pos = np.zeros((self.n, 3), dtype=np.float64)
        
        # Spiral Initialization (Vectorized)
        indices = np.arange(self.n)
        t = indices * 0.5
        self.pos[:, 0] = np.cos(t) * indices * 0.3
        self.pos[:, 1] = np.sin(t) * indices * 0.3
        self.pos[:, 2] = indices * 0.3

    def calculate_energy(self):
        # 1. Distance Matrix (N, N)
        # diffs[i, j] = pos[i] - pos[j]
        # This uses broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diffs = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, np.inf)

        energy = 0.0

        # Hydrophobic Attraction (Only H-H pairs)
        # Create an outer product of the mask to find H-H intersections
        h_interactions = np.outer(self.h_mask, self.h_mask)
        # Keep only upper triangle to avoid double counting
        h_interactions = np.triu(h_interactions, k=1)
        
        # Filter distances where H-H interaction is true and dist < 2.5
        mask = h_interactions & (dists < 2.5)
        if np.any(mask):
            energy -= np.sum(10.0 / (dists[mask] + 0.1))

        # Steric Repulsion (All pairs, non-bonded)
        # Mask out bonded neighbors (k=2 excludes diagonal + 1st off-diagonal)
        non_bonded_mask = np.triu(np.ones((self.n, self.n), dtype=bool), k=2)
        clash_mask = non_bonded_mask & (dists < 0.8)
        
        if np.any(clash_mask):
            # Using the rigid penalty from previous version
            energy += np.sum(20.0 * (1.0 - dists[clash_mask]))

        return energy

    def check_geometry(self):
        # Vectorized check of bond lengths
        bond_vecs = self.pos[1:] - self.pos[:-1]
        bond_dists = np.linalg.norm(bond_vecs, axis=1)
        errors = np.abs(bond_dists - BOND_LENGTH)
        return np.max(errors)

def solve_hyper_fold():
    p = ProteinVectorized(SEQUENCE)
    
    print(f"RUNNING HYPER-FOLD (Vectorized) on {SEQ_NAME}")
    print(f"{'ITER':<6} | {'ENERGY':<10} | {'BOND ERR':<10}")
    print("-" * 45)

    start_time = time.time()
    
    # Pre-allocate useful indices for bond enforcement
    # We enforce bonds separately to keep logic clear, though it could be partially vectorized further
    
    for it in range(MAX_ITER):
        # ----------------------------------------
        # 1. COMPUTE FORCES (Matrix Operation)
        # ----------------------------------------
        forces = np.zeros_like(p.pos)

        # A. Hydrophobic Centroid (The Love Field)
        # Calculate centroid of all H atoms
        h_pos = p.pos[p.h_mask]
        centroid = np.mean(h_pos, axis=0)
        
        # Apply force only to H atoms
        # Vector from Atom to Centroid
        to_center = centroid - p.pos[p.h_mask]
        forces[p.h_mask] += to_center * CENTROID_STRENGTH

        # B. Steric Repulsion (The Fear Field)
        # Pairwise vectors: (N, N, 3)
        diff_tensor = p.pos[:, np.newaxis, :] - p.pos[np.newaxis, :, :]
        # Pairwise distances: (N, N, 1) to match tensor
        dist_matrix = np.linalg.norm(diff_tensor, axis=2)
        # Avoid self-interaction
        np.fill_diagonal(dist_matrix, np.inf) 
        
        # Mask: Neighbors (dist < 1.2) AND Not Bonded (indices diff > 1)
        # Note: Broadcasting the index check is fast
        idx = np.arange(p.n)
        not_bonded = np.abs(idx[:, None] - idx[None, :]) > 1
        
        repulsion_mask = (dist_matrix < STERIC_RADIUS) & not_bonded
        
        # Calculate repulsion only where mask is True
        # F = vec / d^2 * Const
        if np.any(repulsion_mask):
            # Safe division
            d = dist_matrix[repulsion_mask, np.newaxis]
            v = diff_tensor[repulsion_mask]
            
            f_rep = (v / (d**2 + 0.01)) * REPULSION_STRENGTH
            
            # Now we need to accumulate these forces back into the (N, 3) array
            # We iterate over the rows that have repulsions
            # (Fully vectorizing the scatter_add is tricky in pure numpy without ufunc.at, doing loop over N is okay-ish here
            # BUT: We can use np.add.at)
            
            rows, cols = np.where(repulsion_mask)
            np.add.at(forces, rows, f_rep)

        # C. Backbone Springs (The Logic)
        # Bond vectors: (N-1, 3)
        bond_vecs = p.pos[1:] - p.pos[:-1]
        bond_dists = np.linalg.norm(bond_vecs, axis=1)[:, np.newaxis]
        
        # F = k * (dist - L) * (vec / dist)
        spring_f = (bond_dists - BOND_LENGTH) * (bond_vecs / bond_dists) * SPRING_CONST
        
        # Apply to i (forward) and i+1 (backward)
        # forces[i] += spring[i]
        # forces[i+1] -= spring[i]
        forces[:-1] += spring_f
        forces[1:] -= spring_f

        # ----------------------------------------
        # 2. INTEGRATE
        # ----------------------------------------
        
        # Clamp Forces (Safety)
        mags = np.linalg.norm(forces, axis=1)
        # Boolean mask where force is too high
        high_f = mags > 5.0
        if np.any(high_f):
            forces[high_f] *= (5.0 / mags[high_f])[:, np.newaxis]
            
        p.pos += forces * LEARNING_RATE

        # ----------------------------------------
        # 3. ENFORCE BONDS (SHAKE - Vectorized)
        # ----------------------------------------
        # We do 5 passes of parallel correction
        # Note: True parallel update isn't exact for a chain, but converges fast
        for _ in range(5):
            diffs = p.pos[1:] - p.pos[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            errors = dists - BOND_LENGTH
            
            # Identify bad bonds
            mask = np.abs(errors) > 0.01
            if not np.any(mask):
                continue
                
            # Correction vector: delta * (error / dist) * 0.5
            corrections = diffs[mask] * (errors[mask] / dists[mask])[:, np.newaxis] * 0.5
            
            # Apply corrections to p.pos[i] and p.pos[i+1]
            # Since p[i] is involved in bond i-1 and bond i, updating in parallel is approximate
            # We use add.at to handle overlapping indices if needed, or simple slicing
            
            # Indices involved
            idx_i = np.where(mask)[0]
            idx_j = idx_i + 1
            
            p.pos[idx_i] += corrections
            p.pos[idx_j] -= corrections

        if it % 100 == 0:
            e = p.calculate_energy()
            err = p.check_geometry()
            print(f"{it:<6} | {e:<10.2f} | {err:<10.4f}")

    total_time = (time.time() - start_time) * 1000
    final_energy = p.calculate_energy()
    final_err = p.check_geometry()
    
    print("-" * 45)
    print(f"DONE. Time: {total_time:.2f}ms")
    print(f"Final Energy: {final_energy:.2f}")
    print(f"Bond Error:   {final_err:.4f}")

if __name__ == "__main__":
    solve_hyper_fold()