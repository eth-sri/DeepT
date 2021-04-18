import torch

from Models.BERT import BERT


def fgsm_bert(bert_model, example: torch.Tensor, embeddings: torch.Tensor, perturbed_word_index: int, eps: float, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    embeddings_ = embeddings.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    embeddings_.requires_grad_()

    # run the model and obtain the loss
    _, _, info = bert_model.step([example], embeddings=embeddings_, infer_grad=True)
    embeddings_gradient = info["gradients"].clone().detach()

    # Only change the perturbed word
    perturbation = torch.zeros_like(embeddings_gradient)
    perturbation[0, perturbed_word_index] = embeddings_gradient.sign()[0, perturbed_word_index]

    out = embeddings_ + eps * perturbation  # Untargeted attack, move up the error landscape

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def get_odi_starts(bert_model, embeddings, example, eps_odi, perturbed_word_index,
                   output_shape, embeddings_min, embeddings_max):
    w_d = torch.rand(output_shape, device=embeddings.device) * 2 - 1  # rand_like() -> [0, 1)
    for i in range(5):
        w_d_iteration = w_d.clone().detach_()
        embeddings = embeddings.clone().detach_()
        embeddings.requires_grad_()

        _, _, info = bert_model.step(
            [example],
            embeddings=embeddings,
            infer_grad=True,
            custom_loss=lambda logits: w_d_iteration.squeeze().dot(logits.squeeze())
        )

        embeddings_gradient = info["gradients"].clone().detach()
        embeddings_gradient = embeddings_gradient / embeddings_gradient.norm()

        perturbation = torch.zeros_like(embeddings_gradient)
        perturbation[0, perturbed_word_index] = eps_odi * embeddings_gradient.sign()[0, perturbed_word_index]

        # Compute new embeddings before projection
        embeddings = embeddings + perturbation  # Untargeted attack, move up the error landscape

        # Projection Step
        embeddings = torch.max(embeddings_min, embeddings)
        embeddings = torch.min(embeddings_max, embeddings)

    return embeddings


def pgd_bert(bert_model, example, embeddings, perturbed_word_index, num_pgd_starts, pgd_iterations, eps, eps_step, clip_min=None, clip_max=None):
    embeddings_min = embeddings.clone()
    embeddings_max = embeddings.clone()
    embeddings_min[0, perturbed_word_index] -= eps
    embeddings_max[0, perturbed_word_index] += eps

    _, _, info = bert_model.step([example], embeddings=embeddings)
    output_shape = info["pred_scores"].shape

    for j in range(num_pgd_starts):
        if j == 0:
            current_embeddings = embeddings.clone()
        else:
            # Normal perturbation
            perturbation_in_index = torch.rand_like(embeddings[0][perturbed_word_index]) * 2 * eps - eps  # [-eps, eps]
            perturbation = torch.zeros_like(embeddings)
            perturbation[0, perturbed_word_index] = perturbation_in_index
            current_embeddings = embeddings.clone() + perturbation

            use_output_diversified_initialisation = True
            if use_output_diversified_initialisation:
                current_embeddings = get_odi_starts(bert_model, current_embeddings, example, eps_step, perturbed_word_index, output_shape, embeddings_min, embeddings_max)

        for i in range(pgd_iterations):
            # FGSM step
            # We don't clamp here (arguments clip_min=None, clip_max=None)
            # as we want to apply the attack as defined
            current_embeddings = fgsm_bert(bert_model, example, current_embeddings, perturbed_word_index, eps=eps_step)
            # Projection Step
            current_embeddings = torch.max(embeddings_min, current_embeddings)
            current_embeddings = torch.min(embeddings_max, current_embeddings)
        # if desired clip the ouput back to the image domain
        if (clip_min is not None) or (clip_max is not None):
            current_embeddings.clamp_(min=clip_min, max=clip_max)

        _, accurate, _ = bert_model.step([example], embeddings=current_embeddings)
        if not accurate:
            return current_embeddings, True  # True means that PGD attack succesfully worked

    return current_embeddings, False


def pgd_untargeted_bert(bert_model, example, embeddings, perturbed_word_index, num_pgd_starts, pgd_iterations, eps, eps_step, **kwargs):
    return pgd_bert(bert_model, example, embeddings, perturbed_word_index, num_pgd_starts, pgd_iterations, eps, eps_step, **kwargs)


def pgd_attack_bert(bert_model: BERT, args, examples):
    print()
    for sentence_num, example in enumerate(examples):
        # Get embeddings for the sentence
        embeddings, tokens = bert_model.get_embeddings([example])
        embeddings = embeddings if args.cpu else embeddings.cuda()
        length = embeddings.shape[1]

        # [CLS] and [SEP] cannot be perturbed
        for word_num in range(1, length - 1):
            if tokens[0][word_num][0] == "#" or tokens[0][word_num + 1][0] == "#":
                continue

            eps = find_min_eps_for_pgd_attack(bert_model, args, example, embeddings, word_num)
            print("\rSentence {} Word Num {}:    {: <12}  -  Min eps {:.6f}".format(sentence_num, word_num, tokens[0][word_num], eps))


def find_min_eps_for_pgd_attack(bert_model: BERT, args, example, embeddings, word_num):
    left, right = 0.0, args.max_eps
    embeddings = embeddings.clone()  # Ensure we don't mess up the original

    # Test right boundary first
    _, success = pgd_untargeted_bert(
        bert_model, example, embeddings, perturbed_word_index=word_num, num_pgd_starts=args.num_pgd_starts,
        pgd_iterations=args.pgd_iterations, eps=right, eps_step=right * args.eps_step_ratio
    )
    if not success:  # Bound was not high enough, return infinity to indicate we don't know
        return float('inf')

    # If inside the boundaries, use binary search to find the smallest correct eps
    for i in range(15):
        eps = (left + right) / 2
        eps_step = eps * args.eps_step_ratio

        _, success = pgd_untargeted_bert(
            bert_model, example, embeddings, perturbed_word_index=word_num, num_pgd_starts=args.num_pgd_starts,
            pgd_iterations=args.pgd_iterations, eps=eps, eps_step=eps_step
        )

        if success:  # Found an attack for eps, let's see if we can find a smaller one
            right = eps
        else:  # eps wasn't enough, we need to find a higher value
            left = eps

        print("\r{} {:.5f} {:.5f}".format(word_num, left, right), end="")

    return right  # We're sure right leads to an attack, left might not, be prudent and return right
