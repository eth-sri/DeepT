import torch
import torch.nn.functional as F


def fgsm(model, image: torch.Tensor, target: torch.Tensor, eps: float, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    image_ = image.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    image_.requires_grad_()

    # run the model and obtain the loss
    output = F.log_softmax(model(image_), dim=1)
    loss = F.nll_loss(output, target)
    loss.backward()
    gradient = image_.grad.clone().detach()

    # Only change the perturbed word
    perturbation = gradient.sign()

    out = image_ + eps * perturbation  # Untargeted attack, move up the error landscape

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def get_odi_starts(model, image, target, eps_odi,
                   output_shape, image_min, image_max):
    w_d = torch.rand(output_shape, device=image.device) * 2 - 1  # rand_like() -> [0, 1)
    for i in range(5):
        w_d_iteration = w_d.clone().detach_()
        image_ = image.clone().detach_()
        image_.requires_grad_()

        output = F.log_softmax(model(image_), dim=1)
        loss = w_d_iteration.squeeze().dot(output.squeeze())
        loss.backward()
        embeddings_gradient = image_.grad.clone().detach()

        perturbation = eps_odi * embeddings_gradient.sign()

        # Compute new embeddings before projection
        image_ = image_ + perturbation  # Untargeted attack, move up the error landscape

        # Projection Step
        image_ = torch.max(image_min, image_)
        image_ = torch.min(image_max, image_)

    return image_


def pgd(model, image, target, num_pgd_starts, pgd_iterations, eps, eps_step, clip_min=None, clip_max=None):
    image_min = image.clone()
    image_max = image.clone()
    image_min -= eps
    image_max += eps

    output = model(image)
    output_shape = output.shape

    for j in range(num_pgd_starts):
        if j == 0:
            current_image = image.clone()
        else:
            # Normal perturbation
            perturbation = torch.rand_like(image) * 2 * eps - eps  # [-eps, eps]
            current_image = image.clone() + perturbation

            use_output_diversified_initialisation = True
            if use_output_diversified_initialisation:
                current_image = get_odi_starts(model, current_image, target, eps_step, output_shape, image_min, image_max)

        for i in range(pgd_iterations):
            # FGSM step
            # We don't clamp here (arguments clip_min=None, clip_max=None)
            # as we want to apply the attack as defined
            current_image = fgsm(model, current_image, target, eps=eps_step)
            # Projection Step
            current_image = torch.max(image_min, current_image)
            current_image = torch.min(image_max, current_image)

        # if desired clip the ouput back to the image domain
        if (clip_min is not None) or (clip_max is not None):
            current_image.clamp_(min=clip_min, max=clip_max)

        prediction = torch.argmax(model(current_image), dim=-1)
        is_accurate = prediction.eq(target).sum()

        if not is_accurate:
            return current_image, True  # True means that PGD attack succesfully worked

    return current_image, False


def pgd_untargeted(model, image, target, num_pgd_starts, pgd_iterations, eps, eps_step, **kwargs):
    return pgd(model, image, target, num_pgd_starts, pgd_iterations, eps, eps_step, **kwargs)


def pgd_attack(model, args, examples, normalizer):
    print()
    for sentence_num, example in enumerate(examples):
        image, target = example["image"], example["label"]
        image = image if args.cpu else image.cuda()
        eps = find_min_eps_for_pgd_attack(model, args, image, target, normalizer)
        print(f"\rImage {sentence_num + 1}:    Min eps {eps:.6f}")


def find_min_eps_for_pgd_attack(model, args, image, target, normalizer):
    left, right = 0.0, args.max_eps
    image = image.clone()  # Ensure we don't mess up the original

    clip_min, clip_max = [(x - normalizer.mean[0]) / normalizer.std[0] for x in [0.0, 1.0]]

    # Test right boundary first
    _, success = pgd_untargeted(
        model, image, target, num_pgd_starts=args.num_pgd_starts,
        pgd_iterations=args.pgd_iterations, eps=right, eps_step=right * args.eps_step_ratio,
        clip_min=clip_min, clip_max=clip_max
    )
    if not success:  # Bound was not high enough, return infinity to indicate we don't know
        return float('inf')

    # If inside the boundaries, use binary search to find the smallest correct eps
    for i in range(15):
        eps = (left + right) / 2
        eps_step = eps * args.eps_step_ratio

        _, success = pgd_untargeted(
            model, image, target, num_pgd_starts=args.num_pgd_starts,
            pgd_iterations=args.pgd_iterations, eps=eps, eps_step=eps_step,
            clip_min=clip_min, clip_max=clip_max
        )

        if success:  # Found an attack for eps, let's see if we can find a smaller one
            right = eps
        else:  # eps wasn't enough, we need to find a higher value
            left = eps

        left_scaled, right_scaled = left * normalizer.std[0], right * normalizer.std[0]
        print("\r{:.5f} {:.5f}".format(left_scaled, right_scaled), end="")

    return right_scaled  # We're sure right leads to an attack, left might not, be prudent and return right
