import requests
import os
import tempfile
import trimesh

def get_mesh_from_url(mesh_url) -> trimesh.Trimesh:
    """
    Download and load a mesh file from a URL.

    :param mesh_url: URL (str or pydantic Url) pointing to a 3D mesh file
    :return: trimesh.Trimesh object
    :raises Exception: if download or parsing fails
    """
    try:
        mesh_url_str = str(mesh_url)

        response = requests.get(mesh_url_str)
        response.raise_for_status()

        suffix = mesh_url_str.split('?')[0].split('/')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            mesh_path = tmp_file.name

        mesh = trimesh.load(mesh_path, force='mesh')
        os.remove(mesh_path)
        return mesh

    except Exception as e:
        raise RuntimeError(f"Failed to load mesh from URL: {str(e)}")
