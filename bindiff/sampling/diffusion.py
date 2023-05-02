import torch
import sampling.beta_schedule as beta_schedule
from tqdm import tqdm
from einops import repeat


class Diffusion:

    def __init__(
        self,
        beta_small=2e-4,
        beta_large=0.02,
        timesteps=100,
        schedule='linear'
    ):

        self.timesteps = timesteps

        # precompute all betas
        if schedule == 'linear':
            self.betas = beta_schedule.linear_beta_schedule(
                timesteps, beta_small, beta_large
            )

        elif schedule == 'cosine':
            self.betas = beta_schedule.cosine_beta_schedule(timesteps)

        elif schedule == 'quadratic':
            self.betas = beta_schedule.quadratic_beta_schedule(
                timesteps, beta_small, beta_large
            )

        else:
            allowed = beta_schedule.SCHEDULES
            err = f"Schedule must be one of: {allowed}. receieved: {schedule}"
            raise AttributeError(err)

        # precompute all alphas
        a, ac, sqac, sq1ac, pv, sra = beta_schedule.compute_alphas(self.betas)
        self.alphas = a
        self.alphas_cumprod = ac
        self.sqrt_alphas_cumprod = sqac
        self.sqrt_one_minus_alphas_cumprod = sq1ac
        self.posterior_variance = pv
        self.sqrt_recip_alphas = sra

    @staticmethod
    def extract(a, t, x_shape):
        t = t.type(torch.int64)
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out.to(t.device)

    def q_sample(self, x_start, mask, t, noise=None):
        # generate random noise
        if noise is None:
            noise = torch.randn_like(x_start).to(x_start)

        # do not noise context
        mask = mask.unsqueeze(-1)
        noise = noise * mask

        # calculate alpha values for rescaling
        s = x_start.shape

        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, s
        ).to(x_start)

        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, s
        ).to(x_start)

        # rescale noise and input
        scaled_noise = sqrt_one_minus_alphas_cumprod_t * noise
        scaled_input = sqrt_alphas_cumprod_t * x_start
        noised_x = scaled_input + scaled_noise

        # do not noise context
        noised_x = noised_x * mask + x_start * ~mask

        return noised_x, noise

    @torch.no_grad()
    def p_sample(self, model, coords, seqs, masks, t, t_index):
        s = coords.shape

        # extract alhpas
        betas_t = self.extract(self.betas, t, s)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, s
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, s)

        # inference from the model
        _, prediction = model(coords, t, context=seqs, mask=masks)
        # mask = repeat(masks, "b s -> b s c", c=3)
        mask = masks.unsqueeze(-1)
        pred_noise = prediction * mask

        # calculate mean based on the model prediction
        model_mean = sqrt_recip_alphas_t * (
            coords - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, s)
            noise = torch.randn_like(coords)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, coords, seqs, masks, timesteps):
        b = coords.size(0)
        mask = masks.unsqueeze(-1)

        # start with random gaussian noise
        res = torch.randn_like(coords)
        results = [res]

        # iterate over timesteps with p_sample
        desc = 'sampling loop time step'
        for i in tqdm(
            reversed(range(0, timesteps)),
            desc=desc,
            total=timesteps
        ):
            # equal timestep to all samples in batch
            ts = torch.full((b,), i).to(coords)  # all samples same t
            # override with context
            res = res * mask + coords * ~mask
            # forward diffusion
            inference = self.p_sample(model, res, seqs, masks, ts, i)
            results.append(inference)
            res = inference

        return results
