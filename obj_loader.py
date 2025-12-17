import numpy as np
import jax.numpy as jnp

def load_obj(filename: str, scale: float = 1.0, translation: tuple = (0.0, 0.0, 0.0)):
    """
    Loads an OBJ file and returns JAX arrays (v0s, v1s, v2s).
    
    Args:
        filename: Path to .obj file.
        scale: Uniform scaling factor.
        translation: (dx, dy, dz) translation.
        
    Returns:
        v0s, v1s, v2s: (N, 3) JAX arrays representing the mesh triangles.
        center: (3,) JAX array representing the bounding box center of the mesh.
    """
    vertices = []
    # Use a list of lists for faces: [[v0_idx, v1_idx, v2_idx], ...]
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            code = parts[0]
            
            if code == 'v':
                # Vertex: v x y z
                v = [float(x) for x in parts[1:4]]
                vertices.append(v)
            elif code == 'f':
                # Face: f v1/vt/vn v2/vt/vn v3/vt/vn ...
                # OBJ uses 1-based indexing.
                # We only care about vertex indices (first part of split by /)
                face_idxs = []
                for p in parts[1:]:
                    idx_str = p.split('/')[0]
                    face_idxs.append(int(idx_str) - 1)
                
                # Triangulate (Triangle Fan)
                # 0, 1, 2; 0, 2, 3; ...
                v0_idx = face_idxs[0]
                for i in range(1, len(face_idxs) - 1):
                    faces.append([v0_idx, face_idxs[i], face_idxs[i+1]])

    # Convert to Numpy first
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    
    # Apply Transform
    if scale != 1.0:
        vertices *= scale
    
    if translation != (0.0, 0.0, 0.0):
        vertices += np.array(translation, dtype=np.float32)

    # Extract Triangle Vertices
    # Shape: (Num_Triangles, 3)
    # v0s get vertex data for column 0 of faces, etc.
    v0s_np = vertices[faces[:, 0]]
    v1s_np = vertices[faces[:, 1]]
    v2s_np = vertices[faces[:, 2]]
    
    # Convert to JAX
    v0s = jnp.array(v0s_np)
    v1s = jnp.array(v1s_np)
    v2s = jnp.array(v2s_np)
    
    # Calculate center (bounding box center)
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    center_jax = jnp.array(center)
    
    print(f"Loaded {filename}: {len(faces)} triangles.")
    print(f"Center: {center}")
    return v0s, v1s, v2s, center_jax

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        load_obj(sys.argv[1])
