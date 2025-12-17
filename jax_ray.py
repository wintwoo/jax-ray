import jax
import jax.numpy as jnp
from functools import partial

EPSILON = 1e-8

@partial(jax.jit, static_argnames=('M',))
def ray_triangle_intersect_single_ray(orig, direction, v0s, v0v1s, v0v2s, unit_normals, M):
    """
    Kernel for intersecting a single ray with M triangles.
    Uses precomputed edge vectors (v0v1, v0v2) and normals.
    """
    origs = orig[None, :].repeat(M, axis=0)
    dirs = direction[None, :].repeat(M, axis=0)
    
    # [OPTIMIZATION] Use precomputed edges
    # v0v1 = v1s - v0s
    # v0v2 = v2s - v0s
    
    pvec = jnp.cross(dirs, v0v2s)
    det = jnp.sum(v0v1s * pvec, axis=-1)
    parallel_mask = jnp.abs(det) > EPSILON
    
    tvec = origs - v0s
    inv_det = 1.0 / det
    
    u = jnp.sum(tvec * pvec, axis=-1) * inv_det
    u_mask = (u >= 0.0) & (u <= 1.0) & parallel_mask
    
    qvec = jnp.cross(tvec, v0v1s)
    v = jnp.sum(dirs * qvec, axis=-1) * inv_det
    v_mask = (v >= 0.0) & ((u + v) <= 1.0)
    
    t = jnp.sum(v0v2s * qvec, axis=-1) * inv_det
    t_mask = t >= 0.0
    
    intersect_mask = u_mask & v_mask & t_mask
    
    params_result = jnp.stack((t, u, v), axis=-1)
    
    # [OPTIMIZATION] Use precomputed normals
    # unnormalized_normals = jnp.cross(v0v1, v0v2) ...
    # unit_normals passed in
    
    placeholder_params = jnp.full_like(params_result, -1.0)
    placeholder_normals = jnp.full_like(unit_normals, -1.0)
    
    final_params = jnp.where(intersect_mask[:, None], params_result, placeholder_params)
    final_normals = jnp.where(intersect_mask[:, None], unit_normals, placeholder_normals)
    
    return final_params, final_normals, intersect_mask

# --- Pre-computation ---

def precompute_triangle_geometry(v0s, v1s, v2s):
    """
    Precomputes edge vectors and normals for static geometry.
    """
    v0v1s = v1s - v0s
    v0v2s = v2s - v0s
    
    unnormalized_normals = jnp.cross(v0v1s, v0v2s)
    normal_lengths = jnp.linalg.norm(unnormalized_normals, axis=-1, keepdims=True)
    unit_normals = jnp.where(
        normal_lengths > EPSILON,
        unnormalized_normals / normal_lengths,
        jnp.zeros_like(unnormalized_normals)
    )
    return v0v1s, v0v2s, unit_normals

# --- New Reduction Logic ---

def scan_closest_hit_step(carry, batch_data, chunk_size, ray_orig, ray_dir):
    current_min_t, current_normal, current_hit_found = carry
    # Unpack 4 arrays now: v0s, v0v1, v0v2, normals
    v0s_chunk, v0v1_chunk, v0v2_chunk, normals_chunk = batch_data

    params, normals, mask = ray_triangle_intersect_single_ray(
        ray_orig, ray_dir, v0s_chunk, v0v1_chunk, v0v2_chunk, normals_chunk, M=chunk_size
    )
    t_vals = params[..., 0] 

    is_closer = mask & (t_vals < current_min_t)
    candidates_t = jnp.where(is_closer, t_vals, jnp.inf)
    best_idx = jnp.argmin(candidates_t)
    
    chunk_best_t = t_vals[best_idx]
    chunk_best_normal = normals[best_idx]
    chunk_found_better = is_closer[best_idx]

    new_min_t = jnp.where(chunk_found_better, chunk_best_t, current_min_t)
    new_normal = jnp.where(chunk_found_better, chunk_best_normal, current_normal)
    new_hit_found = current_hit_found | chunk_found_better

    return (new_min_t, new_normal, new_hit_found), None

@partial(jax.jit, static_argnames=('M', 'chunk_size'))
def ray_triangle_intersect_closest(
    origs, dirs, 
    v0s, v0v1s, v0v2s, unit_normals, 
    M: int, chunk_size: int
):
    num_chunks = (M + chunk_size - 1) // chunk_size
    padding_needed = num_chunks * chunk_size - M
    pad_width = ((0, padding_needed), (0, 0))
    
    # Pad all geometry arrays
    v0s_batched = jnp.pad(v0s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    v0v1s_batched = jnp.pad(v0v1s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    v0v2s_batched = jnp.pad(v0v2s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    normals_batched = jnp.pad(unit_normals, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    
    triangle_batches = (v0s_batched, v0v1s_batched, v0v2s_batched, normals_batched)

    def per_ray_closest(ray_orig, ray_dir):
        init_carry = (jnp.inf, jnp.zeros(3), False)
        scan_func = partial(scan_closest_hit_step, chunk_size=chunk_size, ray_orig=ray_orig, ray_dir=ray_dir)
        (final_t, final_normal, final_hit), _ = jax.lax.scan(
            scan_func, init=init_carry, xs=triangle_batches
        )
        return final_t, final_normal, final_hit

    return jax.vmap(per_ray_closest)(origs, dirs)

# --- Shadow Logic (Any Hit) ---

def scan_any_hit_step(is_occluded_carry, batch_data, chunk_size, ray_orig, ray_dir, max_dist):
    v0s_chunk, v0v1_chunk, v0v2_chunk, normals_chunk = batch_data
    
    params, _, mask = ray_triangle_intersect_single_ray(
        ray_orig, ray_dir, v0s_chunk, v0v1_chunk, v0v2_chunk, normals_chunk, M=chunk_size
    )
    t_vals = params[..., 0] 

    hit_in_range = mask & (t_vals > 1e-4) & (t_vals < (max_dist - 1e-4))
    chunk_has_hit = jnp.any(hit_in_range)
    new_is_occluded = is_occluded_carry | chunk_has_hit
    return new_is_occluded, None

@partial(jax.jit, static_argnames=('M', 'chunk_size'))
def ray_triangle_intersect_any(
    origs, dirs, max_dists, 
    v0s, v0v1s, v0v2s, unit_normals, 
    M: int, chunk_size: int
):
    num_chunks = (M + chunk_size - 1) // chunk_size
    padding_needed = num_chunks * chunk_size - M
    pad_width = ((0, padding_needed), (0, 0))
    
    v0s_batched = jnp.pad(v0s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    v0v1s_batched = jnp.pad(v0v1s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    v0v2s_batched = jnp.pad(v0v2s, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    normals_batched = jnp.pad(unit_normals, pad_width, mode='edge').reshape(num_chunks, chunk_size, 3)
    
    triangle_batches = (v0s_batched, v0v1s_batched, v0v2s_batched, normals_batched)

    def per_ray_any(ray_orig, ray_dir, max_d):
        init_carry = False 
        scan_func = partial(scan_any_hit_step, chunk_size=chunk_size, ray_orig=ray_orig, ray_dir=ray_dir, max_dist=max_d)
        final_is_occluded, _ = jax.lax.scan(
            scan_func, init=init_carry, xs=triangle_batches
        )
        return final_is_occluded

    return jax.vmap(per_ray_any)(origs, dirs, max_dists)

def create_random_triangles_jax(
    num_triangles: int,
    key: jax.random.PRNGKey,
    min_coord: float = -5.0,
    max_coord: float = 5.0,
    z_plane: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates N random triangles.
    """
    key_v0, key_v1, key_v2 = jax.random.split(key, 3)
    v0s = jax.random.uniform(key_v0, shape=(num_triangles, 3), minval=min_coord, maxval=max_coord)
    v1s = jax.random.uniform(key_v1, shape=(num_triangles, 3), minval=min_coord, maxval=max_coord)
    v2s = jax.random.uniform(key_v2, shape=(num_triangles, 3), minval=min_coord, maxval=max_coord)
    
    if z_plane is not None:
        z_component = jnp.full((num_triangles, 1), z_plane)
        v0s = v0s.at[:, 2].set(z_component[:, 0])
        v1s = v1s.at[:, 2].set(z_component[:, 0])
        v2s = v2s.at[:, 2].set(z_component[:, 0])
    
    return v0s, v1s, v2s

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4)) 
def create_camera_rays_jax(
    width: int = 1024,
    height: int = 1024,
    focal_length: float = 1.0,
    fov_y: float = jnp.deg2rad(90.0),
    samples_per_pixel: int = 1,
    key: jax.random.PRNGKey = None,
    camera_pos: jnp.ndarray = None,
    look_at: jnp.ndarray = None,
    up_dir: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Creates vectorized ray origins and directions.
    """
    plane_half_height = focal_length * jnp.tan(fov_y / 2.0)
    plane_height = 2.0 * plane_half_height
    aspect_ratio = width / height
    plane_width = plane_height * aspect_ratio

    x_indices = jnp.arange(width)
    y_indices = jnp.arange(height)
    X_grid, Y_grid = jnp.meshgrid(x_indices, y_indices) 

    if samples_per_pixel > 1:
        if key is None:
            u_offset = jnp.full((samples_per_pixel, height, width), 0.5)
            v_offset = jnp.full((samples_per_pixel, height, width), 0.5)
        else:
            key_u, key_v = jax.random.split(key)
            u_offset = jax.random.uniform(key_u, (samples_per_pixel, height, width))
            v_offset = jax.random.uniform(key_v, (samples_per_pixel, height, width))
        X_grid = jnp.tile(X_grid[None, ...], (samples_per_pixel, 1, 1))
        Y_grid = jnp.tile(Y_grid[None, ...], (samples_per_pixel, 1, 1))
    else:
        u_offset = jnp.full((1, height, width), 0.5)
        v_offset = jnp.full((1, height, width), 0.5)
        X_grid = X_grid[None, ...]
        Y_grid = Y_grid[None, ...]

    u_coords = (X_grid + u_offset) / width 
    v_coords = (Y_grid + v_offset) / height 

    X_plane = (u_coords - 0.5) * plane_width
    Y_plane = (0.5 - v_coords) * plane_height
    Z_plane = jnp.full_like(X_plane, -focal_length)

    dirs_unnormalized = jnp.stack([X_plane, Y_plane, Z_plane], axis=-1)
    dirs_cam = dirs_unnormalized / jnp.linalg.norm(dirs_unnormalized, axis=-1, keepdims=True)
    
    # 7. Camera Transformation
    if camera_pos is not None and look_at is not None:
        cam_pos = camera_pos
        up = up_dir if up_dir is not None else jnp.array([0.0, 1.0, 0.0])
        
        # Compute camera direction
        camera_dir = look_at - cam_pos
        
        # Build camera basis (right-handed coordinate system)
        forward = camera_dir / jnp.linalg.norm(camera_dir)
        right = jnp.cross(forward, up)  # Changed from cross(up, forward)
        right = right / jnp.linalg.norm(right)
        up_ortho = jnp.cross(right, forward)  # Changed from cross(forward, right)
        
        # Rotation matrix: camera space -> world space
        # Camera looks down -Z, so: [right, up, -forward]
        R = jnp.stack([right, up_ortho, -forward], axis=1)
        
        # Apply rotation to directions
        dirs_flat = dirs_cam.reshape(-1, 3)
        dirs_world = (R @ dirs_flat.T).T
        dirs = dirs_world.reshape(dirs_cam.shape)
        
        # Origins at camera position
        origs = jnp.zeros_like(dirs) + cam_pos
    else:
        # Default: camera at origin, looking down -Z
        dirs = dirs_cam
        origs = jnp.zeros_like(dirs)

    # 8. Flatten
    total_rays = samples_per_pixel * height * width
    origs = origs.reshape(total_rays, 3)
    dirs = dirs.reshape(total_rays, 3)

    return origs, dirs

@jax.jit
def calculate_diffuse_lighting_jax(
    light_positions: jnp.ndarray,
    light_intensities: jnp.ndarray,
    hit_positions: jnp.ndarray,
    unit_normals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates total diffuse lighting (Lambertian).
    """
    num_rays = hit_positions.shape[0]
    num_lights = light_positions.shape[0]

    N = unit_normals[:, None, :] 
    P = hit_positions[:, None, :] 
    L_pos = light_positions[None, :, :]
    L_int = light_intensities[None, :, :]

    L_dir = L_pos - P
    L_norm = jnp.linalg.norm(L_dir, axis=-1, keepdims=True)
    L = L_dir / L_norm 

    N_dot_L = jnp.sum(N * L, axis=-1, keepdims=True)
    diffuse_factor = jnp.maximum(0.0, N_dot_L)
    
    # Attenuation 
    attenuation = 1.0 / (L_norm**2 + 1e-2)
    
    light_contribution = L_int * diffuse_factor * attenuation
    final_diffuse_color = jnp.sum(light_contribution, axis=1) 
    
    return final_diffuse_color

@partial(jax.jit, static_argnums=(6, 7)) # (M, chunk_size)
def calculate_shadow_mask(
    light_positions: jnp.ndarray, 
    hit_positions: jnp.ndarray,
    v0s: jnp.ndarray, 
    v0v1s: jnp.ndarray, v0v2s: jnp.ndarray, unit_normals: jnp.ndarray,
    M: int, chunk_size: int = 128
) -> jnp.ndarray:
    """
    Calculates shadow mask.
    """
    num_rays = hit_positions.shape[0]
    num_lights = light_positions.shape[0]
    
    P = hit_positions[:, None, :] 
    L_pos = light_positions[None, :, :]
    L_vec = L_pos - P 
    L_dist = jnp.linalg.norm(L_vec, axis=-1) 
    L_dir = L_vec / (L_dist[..., None] + 1e-6) 
    
    total_shadow_rays = num_rays * num_lights
    
    shadow_origs = P + L_dir * 1e-3
    origs_flat = shadow_origs.reshape(total_shadow_rays, 3)
    dirs_flat = L_dir.reshape(total_shadow_rays, 3)
    max_dists_flat = L_dist.reshape(total_shadow_rays)
    
    is_occluded_flat = ray_triangle_intersect_any(
        origs_flat, dirs_flat, max_dists_flat,
        v0s, v0v1s, v0v2s, unit_normals, M=M, chunk_size=chunk_size
    )
    
    is_occluded = is_occluded_flat.reshape(num_rays, num_lights)
    visibility_mask = jnp.where(is_occluded, 0.0, 1.0)
    
    return visibility_mask

@partial(jax.jit, static_argnums=(8, 9)) # M, chunk_size
def calculate_lighting_with_shadows_jax(
    light_positions: jnp.ndarray, 
    light_intensities: jnp.ndarray, 
    hit_positions: jnp.ndarray,   
    unit_normals: jnp.ndarray,    
    v0s: jnp.ndarray, 
    v0v1s: jnp.ndarray, v0v2s: jnp.ndarray, mesh_normals: jnp.ndarray, # Precomputed
    M: int, chunk_size: int = 128
) -> jnp.ndarray:
    """
    Calculates Diffuse Lighting + Hard Shadows using precomputed geometry.
    """
    
    visibility = calculate_shadow_mask(
        light_positions, hit_positions,
        v0s, v0v1s, v0v2s, mesh_normals, M, chunk_size
    ) 
    
    N = unit_normals[:, None, :] 
    P = hit_positions[:, None, :] 
    L_pos = light_positions[None, :, :]
    L_int = light_intensities[None, :, :]

    L_dir = L_pos - P
    L_dist = jnp.linalg.norm(L_dir, axis=-1, keepdims=True)
    L = L_dir / (L_dist + 1e-6) 

    N_dot_L = jnp.sum(N * L, axis=-1, keepdims=True) 
    diffuse_factor = jnp.maximum(0.0, N_dot_L)
    
    # Attenuation
    attenuation = 1.0 / (L_dist**2 + 1e-2)
    
    lit_factor = diffuse_factor * visibility[..., None] * attenuation
    
    light_contribution = L_int * lit_factor
    final_diffuse_color = jnp.sum(light_contribution, axis=1) 
    
    return final_diffuse_color