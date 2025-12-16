# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

from lddbm.models.other.lpips import LPIPS
# from lddbm.utils import dist_util
import torchvision.transforms as T
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from PIL import Image

lpips = LPIPS()


# Function to load and preprocess images
def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),  # Convert image to Tensor (C x H x W)
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension (1 x C x H x W)
    return image.to(device)


# Function to calculate PSNR
def calculate_psnr(hr_image, output_image):
    return peak_signal_noise_ratio(output_image, hr_image)


# Function to calculate SSIM
def calculate_ssim(hr_image, output_image):
    return structural_similarity_index_measure(output_image, hr_image)


# Function to calculate LPIPS
def calculate_lpips(lpips_model, hr_image, output_image):
    return lpips_model(output_image, hr_image).mean()


# Main function to compute metrics
def compute_hr_metrics(hr_images, output_images):
    output_images = output_images.to(hr_images.device)
    # PSNR
    psnr = calculate_psnr(hr_images, output_images)

    # SSIM
    ssim = calculate_ssim(hr_images, output_images)

    # LPIPS
    lpips_model = lpips.to(hr_images.device)
    lpips_score = calculate_lpips(lpips_model, hr_images, output_images)

    return psnr, ssim, lpips_score


class SuperResolutionEvaluator:

    def __init__(self):
        super().__init__()

    def run_full_eval(self, model, dl, prefix, save_dir):
        total_psnr, total_ssim, total_lpips_score = 0, 0, 0
        for i, batch in enumerate(dl):
            x, y, _ = batch
            x = x.to(dist_util.dev())
            y = y.to(dist_util.dev())
            x_hat = model.sample(y)
            # calculate super resolution metrics
            psnr, ssim, lpips_score = compute_hr_metrics(x_hat, x)
            total_psnr += psnr
            total_ssim += ssim
            total_lpips_score += lpips_score

        full_psnr = total_psnr / (i + 1)
        full_ssim = total_ssim / (i + 1)
        full_lpips_score = total_lpips_score / (i + 1)

        if prefix is None:
            losses = {'full_psnr': full_psnr, 'full_ssim': full_ssim, 'full_lpips_score': full_lpips_score}

        else:
            losses = {f'{prefix}_full_psnr': full_psnr, f'{prefix}_full_ssim': full_ssim,
                      f'{prefix}_full_lpips_score': full_lpips_score}

        return losses

    def run_one_batch_eval(self, model, batch, prefix):
        x, y, _ = batch
        x = x.to(dist_util.dev())
        y = y.to(dist_util.dev())
        x_hat = model.sample(y)
        # calculate super resolution metrics
        psnr, ssim, lpips_score = compute_hr_metrics(x_hat, x)
        if prefix is None:
            losses = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips_score}
        else:
            losses = {f'{prefix}_psnr': psnr, f'{prefix}_ssim': ssim,
                      f'{prefix}_lpips_score': lpips_score}

        return losses

    def evaluate(self, model, dataloader, eval_type='full', prefix=None, save_dir=''):
        assert eval_type in ['full', 'one_batch'], "Can run evaulation on full dataloader or batch only."
        model.eval()

        if eval_type == 'full':
            metrics_dict = self.run_full_eval(model, dataloader, prefix, save_dir)

        else:
            batch = next(iter(dataloader))
            metrics_dict = self.run_one_batch_eval(model, batch, prefix)

        return metrics_dict
