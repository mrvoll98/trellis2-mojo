# Image conditioning (WP9 part 3 step 3; pure Mojo since WP13+WP14).
#
# ImageConditioner is the ONLY thing the runner sees: image path in, dense
# cond tensor [1, L, 1024] out. The whole path is pure Mojo: PAM/PPM
# decode (io/image.mojo), preprocess + Lanczos + ImageNet normalization
# (imaging/, bit-identical to the PIL originals) and the DINOv3 ViT
# (models/dinov3.mojo, weights via the WP12 safetensors reader).
#
# Input must be a PAM P7 file, RGBA with a real alpha channel (ADR 0007 —
# the rembg/BiRefNet path is not supported). PNG -> PAM conversion is
# documented in README_MOJO.md.

from trellis2_mojo.checkpoints import load_dinov3
from trellis2_mojo.gpu.linear import GpuContext
from trellis2_mojo.imaging.preprocess import preprocess_rgba, cond_pixels
from trellis2_mojo.io.image import read_image
from trellis2_mojo.models.dinov3 import Dinov3ViT
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


struct ImageConditioner(Movable):
    var model: Dinov3ViT

    def __init__(out self, gpu: Optional[GpuContext] = None) raises:
        self.model = load_dinov3(gpu)

    def get_cond(self, image_path: String, resolution: Int) raises -> Tensor[F32]:
        """Preprocess (alpha crop/premultiply) + DINOv3 features [1, L, 1024]."""
        var pre = preprocess_rgba(read_image(image_path))
        return self.model.forward(cond_pixels(pre, resolution))


def zeros_like_cond(cond: Tensor[F32]) raises -> Tensor[F32]:
    """Create the pipeline's zero-valued negative conditioning tensor."""
    return Tensor[F32](cond.shape.copy())
