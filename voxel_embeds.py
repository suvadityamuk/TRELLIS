import os
import json
import hashlib
import open3d as o3d
import trimesh
from easydict import EasyDict as edict
from subprocess import DEVNULL, call
import numpy as np
import torch
import utils3d
from dataset_toolkits.utils import sphere_hammersley_sequence
import trellis.models as models
import trellis.modules.sparse as sp

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

encoder = models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16').eval().cuda()

def convert_glb_to_ply(input_glb_path, output_ply_path):
    try:
        # Load the GLB file
        scene = trimesh.load(input_glb_path)

        # If the GLB contains multiple meshes, you might want to combine them
        # or save them individually. For simplicity, this example assumes
        # a single mesh or combines all meshes into one.
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene # It's already a single mesh

        # Export the mesh to PLY format
        mesh.export(output_ply_path, file_type='ply')
        print(f"Successfully converted '{input_glb_path}' to '{output_ply_path}'")

    except Exception as e:
        print(f"Error during conversion: {e}")

def get_sha256_hash_string(input_string):
    """Generates the SHA-256 hash of a given string."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8')) # Encode the string to bytes
    return sha256_hash.hexdigest()

def _voxelize(file, sha256, output_dir):
    # mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    mesh = o3d.io.read_triangle_mesh(file)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}

def get_voxels(input_dir, sha256, resolution):
    position = utils3d.io.read_ply(os.path.join(input_dir, 'voxels', f'{sha256}.ply'))[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss

def _extract_dino_features(input_dir, sha256, save_path, resolution=64):
    ss = get_voxels(input_dir, sha256, resolution)[None].float()
    latent = encoder(ss, sample_posterior=False)
    latent_np = latent[0].cpu().numpy()
    np.savez_compressed(save_path, mean=latent_np)
    return {'sha256': sha256, 'encoded': True}

def _encode_slat_latents(feature_npz_path, encoder_ckpt):
    feats = np.load(feature_npz_path)
    patchtokens = torch.from_numpy(feats['patchtokens']).float()
    indices = torch.from_numpy(feats['indices']).int()
    # Construct SparseTensor
    coords = torch.cat([torch.zeros(patchtokens.shape[0], 1).int(), indices], dim=1)
    sparse_tensor = sp.SparseTensor(feats=patchtokens, coords=coords).cuda()
    # Load encoder
    encoder = models.from_pretrained(encoder_ckpt).eval().cuda()
    # Encode
    with torch.no_grad():
        latent = encoder(sparse_tensor, sample_posterior=False)
    # latent.feats is the latent embedding
    np.savez_compressed('/path/to/save/latent.npz', **latent)
    return {
        'feats': latent.feats.cpu().numpy().astype(np.float32),
        'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
    }

def _encode_partfield_latents(mesh_path, encoder_ckpt='microsoft/TRELLIS-image-large/ckpts/partfield_enc'):
    """Encode mesh using Partfield for part segmentation"""
    from partfield.model.PVCNN.pc_encoder import PCEncoder
    from partfield.dataloader import load_mesh_pc
    
    # Load point cloud from mesh
    points = load_mesh_pc(mesh_path)
    
    # Initialize Partfield encoder
    encoder = PCEncoder()
    encoder.load_state_dict(torch.load(encoder_ckpt))
    encoder = encoder.cuda().eval()
    
    # Get Partfield latents 
    with torch.no_grad():
        points = torch.FloatTensor(points).cuda().unsqueeze(0)
        latents = encoder(points)
        
    return {
        'latents': latents.cpu().numpy(),
        'points': points.cpu().numpy()
    }
    

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, num_views):
    # output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path), 
        '--resolution', '512',
        '--output_folder', output_dir,
        '--engine', 'CYCLES',
        # '--engine', 'BLENDER_EEVEE',
        '--save_mesh',
        '--noaudio',
        '--t 1'

    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    call(args)
    
    if os.path.exists(os.path.join(output_dir, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}
    
def main():
    # Get path to GLB file
    path = os.path.join(os.getcwd(), "sofa.glb")
    print(f"Processing file: {path}")

    # Convert GLB to PLY
    ply_path = os.path.join(os.getcwd(), "sofa_converted.ply")
    convert_glb_to_ply(path, ply_path)

    # Install Blender if not available
    _install_blender()

    # Set up parameters
    sha256 = get_sha256_hash_string(ply_path)
    render_output_dir = os.path.join(os.getcwd(), "guideflow_render_output", 'renders', sha256)
    os.makedirs(render_output_dir, exist_ok=True)
    num_views = 150

    # Call _render function to get images of views
    _render(path, sha256, render_output_dir, num_views)
    print(f"Rendered images saved in {render_output_dir}")
    print(f"Copying PLY file to render directory.")
    
    # Copy the converted PLY file to the render directory
    os.system(f"cp {ply_path} {os.path.join(render_output_dir, 'mesh.ply')}")
    print(f"PLY file copied to {os.path.join(render_output_dir, 'mesh.ply')}")
    
    # Use voxelize to create voxel representation
    voxel_output_dir = os.path.join(os.getcwd(), "guideflow_voxel_output")
    os.makedirs(os.path.join(voxel_output_dir, 'voxels'), exist_ok=True)
    
    # Save voxels and metadata
    _voxelize(ply_path, sha256, voxel_output_dir)
    print(f"Rendered images and voxel data saved in {render_output_dir} and {voxel_output_dir} respectively.")

    # Run DINOv2 to get features for downstream SLat VAE
    print("Running DINOv2 to extract features...")
    _extract_dino_features(voxel_output_dir, sha256, os.path.join(voxel_output_dir, 'ss_latents', f'{sha256}.npz'))
    print(f"DINOv2 features saved in {os.path.join(voxel_output_dir, 'ss_latents', f'{sha256}.npz')}")

    # Encode SLat VAE latents with TRELLIS encoder
    print("Encoding SLat VAE latents with TRELLIS encoder...")
    _encode_slat_latents(os.path.join(voxel_output_dir, 'ss_latents', f'{sha256}.npz'), 'microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    print("SLat VAE latents encoded.")

    # Encode mesh with Partfield
    print("Encoding mesh with Partfield...")
    partfield_latents = _encode_partfield_latents(ply_path)

    # Save Partfield latents
    partfield_output_dir = os.path.join(os.getcwd(), "guideflow_partfield_output")
    os.makedirs(partfield_output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(partfield_output_dir, f'{sha256}_partfield.npz'),
        **partfield_latents
    )
    print(f"Partfield latents saved in {partfield_output_dir}")

if __name__ == "__main__":
    main()