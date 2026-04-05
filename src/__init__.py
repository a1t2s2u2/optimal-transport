from .data import load_dataset, image_to_dist, build_pixel_cost, downsample
from .sinkhorn import sinkhorn_log, sinkhorn_batch, sinkhorn_loss_batch, sinkhorn_value_log
from .autoencoder import SinkhornAutoencoder
from .cuturi import benchmark_sinkhorn_scaling, smoothness_profile
