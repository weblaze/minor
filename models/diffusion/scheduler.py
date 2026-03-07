import torch
import torch.nn.functional as F

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Pre-calculate alphas and betas
        # Linear schedule used in DDPM
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate standard deviation for q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def set_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward process: q(x_t | x_0)
        Adds noise to the original samples to create noisy samples at time t.
        """
        # Ensure standard deviation shapes match the batch dimension
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape to allow broadcasting with original samples (Batch, C, H, W)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        Backward process: q(x_{t-1} | x_t, x_0)
        Computes the previous sample x_{t-1} from the current sample x_t and predicted noise.
        """
        t = timestep
        prev_t = t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 1. Compute predicted original sample (x_0 estimation)
        # x_0 = (x_t - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        # 2. Compute coefficients for x_0 and x_t
        # Sample x_{t-1} from q(x_{t-1} | x_t, x_0)
        pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = (torch.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t

        # 3. Compute predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 4. Add noise (variance)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            # Use fixed-large variance (simplified)
            variance = torch.sqrt(current_beta_t) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
