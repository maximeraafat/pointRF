import torch

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# point renderer
def point_renderer(cameras, image_size, radius=0.005, points_per_pixel=50, background=[0, 0, 0]):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )

    rasterizer = PointsRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background)
    )

    return renderer.to(device)