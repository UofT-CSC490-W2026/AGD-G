# Adversarial-Guided Diffusion

Implementation of the method described by [Lin et al, 2025][1].

The method injects a *target* image into the diffusion process on a *clean* image.
The resulting *adversarial* image will look visually like the *clean* image, but
computer vision models, including those used in multimodal LLMs, will classify
the image as similar to the *target* image.

The implementation is based on the following [pseudocode from the paper][1].
Here, we've adapted the mathematical notation to plaintext and added
comments, most from the surrounding paragraphs in the paper.
Line numbers are preserved to match the pseudocode in the paper.

```
Algorithm 1: Adversarial image generation algorithm.
----------------------------------------------------
Input:
- a^tar,             # target text
- epsilon_theta(x),  # predicted noise for reverse diffusion step
- Delta,             # step index in {T...1} to start injecting target image
- x^cle_0;           # clean image

    # Generate target image with a text-to-image model
1:  x^tar_0 = T2I(a^tar);
    # Generate CLIP feature vector for target image
2:  z^tar_0 = CLIP(x^tar_0);
    # Add lots of noise to the clean image
3:  Obtain x^cle_T = (
            sqrt(alpha_bar_T) * x^cle_0
            + sqrt(1-alpha_bar_T) * epsilon
        ), epsilon ~ Norm(0, I);
4:  for t in {T, ..., 2, 1} do
5:      if t > Delta then
            # For the first T - Delta steps, perform normal reverse diffusion
6:          x^cle_{t-1} = denoise(x^cle_t, epsilon_theta(x^cle_t));
7:      else
            # For the last Delta steps, inject target image's caption
8:          if t == Delta then
9:              x_t = x^cle_t;
10:         end if
11:         x_tilde = x_t;
            # Predict noise to be subtracted (clean reverse diffusion)
            epsilon = epsilon_theta(x_t);
            epsilon_bar = 0;
            # Generate target noise signal epsilon^tar
12:         for tau in {N, ..., 2, 1} do
13:             epsilon = epsilon + gamma * sign(epsilon_bar);
                # Compute moving average of epsilon
                # Note epsilon^tar is not defined yet; this is a bug in the paper
                # This can be moved to after line 17 to fix it
14:             epsilon_bar = lambda * epsilon_bar + (1 - lambda) * epsilon^tar;
                # Reverse diffusion step
15:             x_tilde = denoise(x_tilde, epsilon);
16:             z = CLIP(x_tilde);
                # Compute deviation of z from target z
17:             epsilon^tar = -sqrt(1 - alpha_bar_T)
                    * gradient(Norm(
                        z | sqrt(alpha_bar_T) * z^tar_0,
                        (1 - alpha_bar_T) * I
                    ));
18:         end for
            # Combine clean noise and target noise (with strength gamma)
19:         epsilon_tilde = epsilon_theta(x_t) + gamma * sign(epsilon_bar);
            # Reverse diffusion step with adversarial noise
20:         x_{t-1} = denoise(x_t, epsilon_tilde);
21:     end if
22: end for
Output: An adversarial image: x_0.
```

In this implementation, we skip the image generation step (line 1) and take
the target image as input instead.

Stable Diffusion implements:
- Line 3 as `scheduler.add_noise(x^cle_0, epsilon, 0)`
- `denoise(x_t, epsilon)` as `scheduler.step(epsilon, x_t, x_t)`
- `epsilon_theta(x_t)` as `unet(x_t, t, encoder_hidden_states=prompt)`
- `alpha_bar_t` as `scheduler.alphas_cumprod[t]` (SD handles the alpha-scheduling)

May have to do `CLIP(vae.decode(x_tilde))` instead of just `CLIP(x_tilde)` because
`x_tilde` is a vector in Stable Diffusion's latent space, not in pixel space.

[1]: Guang Lin and Zerui Tao and Jianhai Zhang and Toshihisa Tanaka and Qibin Zhao.
        Adversarial Guided Diffusion Models for Adversarial Purification.
        arXiv preprint arXiv:2403.16067, 2025.
