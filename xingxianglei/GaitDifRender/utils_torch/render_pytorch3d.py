import numpy as np
import pytorch3d
import pytorch3d.renderer
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation


def render_mesh(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    vertices = vertices + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    faces = faces.expand(bs, *faces.shape).to(device)

    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),  # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs


def render_silhouette_mesh(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    vertices = vertices + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    faces = faces.expand(bs, *faces.shape).to(device)

    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

    # Initialize each vertex to be white in color.
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    sigma = 1e-4
    raster_settings_silhouette = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=75
    )

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_silhouette
        ),
        shader=pytorch3d.renderer.SoftSilhouetteShader()
    )

    # Do rendering
    imgs = renderer(mesh)

    # binary silhouettes
    def custom_function(x, threshold=1e-5):
        return torch.sigmoid((x - threshold) * 10)

    # silhouette_binary = (imgs[..., 3] > 1e-4).float()
    # silhouette_binary = custom_function(imgs[..., 3])
    silhouette_binary = imgs[..., 3].float()
    # sils = imgs[..., [3]]
    return silhouette_binary


class SilhouetteRenderer(nn.Module):
    def __init__(self,
                 focal_length=1000.0,
                 height=128,
                 width=128,
                 device=None):
        super().__init__()
        # Initialize a camera.
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
            device=device,
        )
        # Define the settings for rasterization and shading.
        sigma = 5e-5
        raster_settings_silhouette = pytorch3d.renderer.RasterizationSettings(
            image_size=(height, width),
            blur_radius=0,
            faces_per_pixel=1,
        )

        # Silhouette renderer
        self.mesh_render = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                raster_settings=raster_settings_silhouette
            ),
            shader=pytorch3d.renderer.SoftSilhouetteShader()
        )

    def forward(self, vertices, faces, translation, device=None):
        ''' Render the mesh under camera coordinates
          vertices: (N_v, 3), vertices of mesh
          faces: (N_f, 3), faces of mesh
          translation: (3, ), translations of mesh or camera
        '''
        if device is None:
            device = vertices.device
        batch_size = vertices.shape[0]

        # add the translation
        vertices = vertices + translation[:, None, :]

        # upside down the mesh
        # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
        rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
        faces = faces.expand(batch_size, *faces.shape).to(device)

        vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)
        # vertices = RotateAxisAngle(180, 'X').to(device).transform_points(vertices)

        vertices = RotateAxisAngle(180, 'Y').to(device).transform_points(vertices)

        mesh = Meshes(verts=vertices, faces=self.faces.expand(batch_size, -1, -1).cuda())  # 构建mesh

        # 设置相机参数
        cameras = get_camera(camera_param, device=vertices.device, dtype=vertices.dtype)
        # plot_scene_mesh(mesh=mesh[0], cameras=cameras[0])
        silhouette = self.mesh_render(meshes_world=mesh, cameras=selcameras)
        silhouette = silhouette[..., 3]

        return silhouette


def render_mesh_single_frame(vertices, faces, translation, focal_length, height, width, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    assert vertices.shape[0] == 1
    vertices = vertices[0]
    translation = translation[0]

    # add the translation
    vertices = vertices + translation

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device)
    faces = faces.to(device)

    vertices = torch.matmul(rot, vertices.T).T

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices], faces=[faces], textures=textures)

    # Initialize a camera.
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=((2 * focal_length / min(height, width),
                       2 * focal_length / min(height, width)),),
        device=device,
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        # image_size=(height, width),   # (H, W)
        image_size=height,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(
        device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs[0]
