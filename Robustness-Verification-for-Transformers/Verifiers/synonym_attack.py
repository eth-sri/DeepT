import math
#from gurobipy import Model, GRB, LinExpr, quicksum

import torch
from typing import List, Tuple, Mapping, Union

from scipy.special import loggamma

from Models.BERT import BERT

# Setup the synonyms dictionary

# with open('Verifiers/synonyms.json') as synonyms_file:
#     synonyms: Mapping[str, int] = json.load(synonyms_file)


def get_norm(tensor: torch.Tensor, p: float, dim: Union[int, List[int]]) -> torch.Tensor:
    return torch.linalg.norm(tensor, ord=p, dim=dim)

# len(synonyms)
# >>> 49975
#
# c = Counter([len(v) for v in synonyms.values()])
# >>> Counter({0: 24384,
#              8: 6253,
#              2: 3844,
#              3: 3081,
#              7: 1224,
#              4: 2388,
#              1: 5382,
#              5: 1867,
#              6: 1552}
# Most words don't have synonyms


def build_vocab_dict(model: BERT):
    return dict(model.vocab)


def get_valid_neighbors(example: Mapping, token: str, token_pos: int, vocab_dict) -> List[str]:
    """
    Returns the synonyms of the word which are part of the vocabulary and are not a special token.
    These synonyms also have a good language property, e.g. they don't modify the sentence meaning too much.
    We rely on the pre-processeded file made available in the Lirpa paper.
    TODO: vocabulary is a bit different, have to adjust this so it's more comparable.
    """
    candidate_neighbors = example['candidates'][token_pos]
    return [neighbor for neighbor in candidate_neighbors if neighbor in vocab_dict and neighbor[0] != '[']


def get_word_embeddings_for_token_and_its_synonyms(example: Mapping, token: str, token_pos: int, model: BERT,
                                                   vocab_dict, max_synonyms_per_word: int) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Finds the N synonyms of the token and creates a (1 + N) x E tensor containing the embeddings of the token
    and its synonyms, where each embedding for a token is stored in a different row
    Args:
        example: the dictionary representing the sentence and label
        token: the token whose synonyms we search for
        token_pos: position of the token in the sentence
        model: the model that provides the embeddings
        vocab_dict: vocabulary
        max_synonyms_per_word: number of substitutes allowed per word
    """

    # substitution_budget = 2
    if token[0] == '[':  # or token_pos > 6:
        all_tokens = [token]
    else:
        synonyms_for_token = get_valid_neighbors(example, token, token_pos, vocab_dict)
        good_synonyms = [x for x in synonyms_for_token[:max_synonyms_per_word] if token != x]
        all_tokens = [token] + good_synonyms

    example_with_synonyms = {'label': example['label'], 'sent_a': all_tokens, 'sentence': ' '.join(all_tokens)}
    all_word_embeddings, _ = model.get_embeddings(batch=[example_with_synonyms], only_word_embeddings=True)

    all_word_embeddings = all_word_embeddings[0, 1:1 + len(all_tokens)]

    # A [CLS] token was added at the beginning and a [SEP] token at the end. Don't get the embeddings for these
    num_neighbors = len(all_tokens) - 1
    return all_word_embeddings, num_neighbors, all_tokens

    # num_neighbors = all_word_embeddings.shape[0] - 1
    # if all_word_embeddings.shape[0] > 2:
    #     changes = (all_word_embeddings[0] - all_word_embeddings[1:]).abs().mean(dim=1)
    #     closest_word_embedding = changes.argmin()
    #     result = torch.cat([
    #         all_word_embeddings[0:1],
    #         all_word_embeddings[1+closest_word_embedding:1+closest_word_embedding + 1]
    #     ], dim=0)
    #
    #     all_word_embeddings = result
    #     num_neighbors = 1

    # good = torch.logical_and((changes < 0.005), (changes > 0))


def get_word_embeddings_region_for_token_and_synonyms(example: Mapping, token: str, token_pos: int, model: BERT, vocab_dict, max_synonyms_per_word: int) -> Tuple[torch.Tensor, torch.Tensor, int, List[str]]:
    """
    Returns the lower and upper bounds for the embedding region containing the embeddings of the token and its synonyms.
    The returned tensor has shape (1, E), where E is the embedding size.
    """
    all_word_embeddings, num_neighbors, all_words = get_word_embeddings_for_token_and_its_synonyms(example, token, token_pos, model, vocab_dict, max_synonyms_per_word)
    lower, _ = all_word_embeddings.min(dim=0, keepdim=True)
    upper, _ = all_word_embeddings.max(dim=0, keepdim=True)
    return lower, upper, num_neighbors, all_words


def get_embeddings_region_sentence_with_synonyms(example: Mapping, model: BERT, max_synonyms_per_word: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, List[List[str]]]:
    """
    Given a sentence, for each token we find the word embedding of the token and its synonyms and then define a word embedding
    region containing these embeddings. The position and token embeddings are then added to that. The final region contains
    the embeddings of the sentence where each word could be replaced by its adequate synonyms

    Args:
        example: the dictionary representing the sentence and label
        model: the BERT model which provides the embeddings
        max_synonyms_per_word: number of substitutes allowed per word

    Returns:
        The lower and upper bounds of the embedding regions. More precisely, two tensors of shape (N, E) are returned
        where N is the number of token and E is the embedding size. We also return the number of neighbors.
    """
    vocab_dict = build_vocab_dict(model)

    tokens = ["[CLS]"] + example['sent_a'] + ["[SEP]"]
    position_and_token_embeddings, _ = model.get_embeddings([example], only_position_and_token_embeddings=True, custom_seq_length=len(tokens))
    position_and_token_embeddings = position_and_token_embeddings[0]
    lowers, uppers = [], []
    candidate_words_per_position = []
    total_num_neighbor_words = 0
    total_num_neighbor_combinations = 1
    for i, token in enumerate(tokens):
        token_pos = i - 1  # Because get_embeddings adds a "[CLS]" token at the beginning and a "[SEP]" at the end
        word_embedding_lower, word_embedding_upper, num_neighbors, all_words = get_word_embeddings_region_for_token_and_synonyms(example, token, token_pos, model, vocab_dict, max_synonyms_per_word)
        candidate_words_per_position.append(all_words)
        embedding_lower = word_embedding_lower + position_and_token_embeddings[i]
        embedding_upper = word_embedding_upper + position_and_token_embeddings[i]
        lowers.append(embedding_lower)
        uppers.append(embedding_upper)
        total_num_neighbor_words += num_neighbors
        total_num_neighbor_combinations *= (1 + num_neighbors)

    sentence_embeddings_lower = torch.cat(lowers)
    sentence_embeddings_upper = torch.cat(uppers)
    return sentence_embeddings_lower, sentence_embeddings_upper, total_num_neighbor_words, total_num_neighbor_combinations, candidate_words_per_position


def get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms(example: Mapping, model: BERT, max_synonyms_per_word: int) -> Tuple[torch.Tensor, int, int, List[List[str]], torch.Tensor, torch.Tensor]:
    """
    Creates the weights of a zonotope representing the region that contains the embeddings of each token and their synonyms.
    This simulates a synonym-substitution attack on a text, where words can be substituted by any synonym.
    Args:
        example: the dictionary representing the sentence and label
        model: the BERT model that provides embeddings
        max_synonyms_per_word: number of substitutes allowed per word

    Returns:
        The zonotope weights representing the regions of the embeddings of the tokens in the sentence and their synonyms.
        These weights have shape (1 + M, N, E) where M <= N * E. M = N * E if all words can be perturbed (aka have synonyms)
        otherwise M can be lower than N * E since there are words where there is no uncertainty. We also return
        the number of neighbors.
    """
    sentence_embeddings_lower, sentence_embeddings_upper, number_neighbors, num_enumerations, candidate_words_per_position = get_embeddings_region_sentence_with_synonyms(example, model, max_synonyms_per_word)
    assert (sentence_embeddings_lower <= sentence_embeddings_upper).all(), \
        "Sentence embedding lower bounds are sometimes bigger than the upper bounds"

    center = 0.5 * (sentence_embeddings_lower + sentence_embeddings_upper)
    radius = 0.5 * (sentence_embeddings_upper - sentence_embeddings_lower)

    # volume = (2 * radius).prod()
    # log_volume = (2 * radius[radius > 0]).log().sum()
    # print(f"Log Volume: {log_volume}")


    # TODO: implement a version where terms without uncertainty have no error terms
    #       (to make the zonotope a bit smaller and make operations faster)

    # Create zonotope weights
    N, E = sentence_embeddings_lower.shape
    zonotope_weights = torch.zeros(1 + N * E, N, E, device=model.device)

    # Fill center
    zonotope_weights[0] = center

    # Fill error terms
    indices = torch.arange(1, 1 + N * E, device=model.device)
    has_new_error_terms = torch.ones_like(radius, dtype=torch.bool, device=model.device)
    zonotope_weights[indices, has_new_error_terms] = radius[has_new_error_terms]

    return zonotope_weights, number_neighbors, num_enumerations, candidate_words_per_position, center, radius


def get_embeddings_per_word(example: Mapping, model: BERT, max_synonyms_per_word: int) -> Tuple[List[torch.Tensor], int, int, List[List[str]]]:
    """
    Given a sentence, for each token we find the word embedding of the token and its synonyms and then define a word embedding
    region containing these embeddings. The position and token embeddings are then added to that. The final region contains
    the embeddings of the sentence where each word could be replaced by its adequate synonyms

    Args:
        example: the dictionary representing the sentence and label
        model: the BERT model which provides the embeddings
        max_synonyms_per_word: number of substitutes allowed per word

    Returns:
        The lower and upper bounds of the embedding regions. More precisely, two tensors of shape (N, E) are returned
        where N is the number of token and E is the embedding size. We also return the number of neighbors.
    """
    vocab_dict = build_vocab_dict(model)
    tokens = ["[CLS]"] + example['sent_a'] + ["[SEP]"]
    position_and_token_embeddings, _ = model.get_embeddings([example], only_position_and_token_embeddings=True, custom_seq_length=len(tokens))
    position_and_token_embeddings = position_and_token_embeddings[0]
    embeddings_per_word = []
    candidate_words_per_position = []
    total_num_neighbor_words = 0
    total_num_neighbor_combinations = 1
    for i, token in enumerate(tokens):
        token_pos = i - 1  # Because get_embeddings adds a "[CLS]" token at the beginning and a "[SEP]" at the end
        all_word_embeddings, num_neighbors, all_tokens = get_word_embeddings_for_token_and_its_synonyms(example, token, token_pos, model, vocab_dict, max_synonyms_per_word)
        complete_embeddings_for_word = all_word_embeddings + position_and_token_embeddings[i]
        embeddings_per_word.append(complete_embeddings_for_word)
        candidate_words_per_position.append(all_tokens)
        total_num_neighbor_words += num_neighbors
        total_num_neighbor_combinations *= (1 + num_neighbors)

    return embeddings_per_word, total_num_neighbor_words, total_num_neighbor_combinations, candidate_words_per_position


def get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms_l_norm(example: Mapping, model: BERT, max_synonyms_per_word: int, p=2, use_solver=False) -> Tuple[torch.Tensor, int, int, List[List[str]]]:
    """
    Creates the weights of a zonotope representing the region that contains the embeddings of each token and their synonyms.
    This simulates a synonym-substitution attack on a text, where words can be substituted by any synonym.
    Args:
        example: the dictionary representing the sentence and label
        model: the BERT model that provides embeddings
        max_synonyms_per_word: number of substitutes allowed per word

    Returns:
        The zonotope weights representing the regions of the embeddings of the tokens in the sentence and their synonyms.
        These weights have shape (1 + M, N, E) where M <= N * E. M = N * E if all words can be perturbed (aka have synonyms)
        otherwise M can be lower than N * E since there are words where there is no uncertainty. We also return
        the number of neighbors.
    """
    assert p == 1 or p == 2 or p > 10, "Norm must be 1 or 2 or inf (> 10)"
    embeddings_per_word, number_neighbors, num_enumerations, candidate_words_per_position = get_embeddings_per_word(example, model, max_synonyms_per_word)
    # embeddings_per_word is a N-element list, where every element is a (1 + Num_synonyms_for_ith_word) x E
    # E is the size of an embedding (not the number of error terms)

    # Create zonotope weights
    N = len(embeddings_per_word)
    E = embeddings_per_word[0].size(1)
    zonotope_weights = torch.zeros(1 + N * E, N, E, device=model.device)

    error_index = 1
    log_volume = 1.0
    for word_num in range(N):
        # Imagine for the k-th token, there's a single embedding at a distance (2, -3, 4) from the center
        # the distance will be sqrt(2² + (-3)² + 4²) = sqrt(29)
        # Then, if we assume that the center is (0, 0, 0), the zonotope for the embedding of token will be:
        #   x1 = sqrt(29) e1
        #   x2 =                sqrt(29) e2
        #   x3 =                                 sqrt(29 / 3) e3
        #   with sqrt(e1² + e2² + e3²) <= 1
        # Do these constraints ensure that the embedding region is correctly captured as a L2 convex region?
        # To determine that, we need to ensure that all the embeddings of the word and its substitutes are within the
        # L2-norm zonotope defined above.
        #
        # Key idea: Since p = 2, then q = 2, so the dual norm logic is particularly simple for this
        # case, we can simply re-use the 2-norm to compute the maximum boundaries:
        #   - What are the concrete lower and upper bounds?
        #       U1 = 2-norm(sqrt(29 / 3), 0, 0)) = sqrt(29)
        #       U2 = 2-norm(sqrt(29 / 3), 0, 0)) = sqrt(29)
        #       U3 = 2-norm(sqrt(29 / 3), 0, 0)) = sqrt(29)
        #   So we have for each variable x_i for -sqrt(29) <= x_i <= sqrt(29)
        #   Or more generally,  -max_dist <= x_i <= max_dist
        #
        #   - Does this imply that all valid embeddings P in S are contained in the zonotope? Let's define some terminology:
        #      S = set of embeddings for all the points
        #      P = embedding in S we're currently considering
        #      M = embedding at the maximal distance D from the origin
        #   By definition we know that dist(P, origin) <= D, because we defined M to be the most distant point.
        #   Will S be inside the bounds? To find that out, we must discover whether there's an assignment to the
        #   noise symbols that makes the variables equal S (x1=s1, ..., xn=sn).
        #
        #   x1 = c1 e1             =>   e1 = x1 / c1
        #   ...
        #   xn = cn en             =>   en = xn / cn
        #
        #   and we must respect 2-norm(e1, ..., en) <= 1
        #   that is             e1² + ... + en² <= 1
        #                      (x1/c1)² + ... + (xn/cn)² <= 1
        #   If we pick, c1 = c2 = ... = cn = D, then we have that
        #   x1² + ... + xn² <= D². Is this true for all S? Well yes, by definition, since we picked
        #   D = max(P) 2-norm(P) = max(P) 2-norm(p1, ..., pn) >= 2-norm(x1, ..., xn)
        #   and therefore D² >= x1² + ... + xn²
        # Variant1:

        all_embeddings_for_position = embeddings_per_word[word_num].unique(dim=0)  # (1 + NumSynonyms, E)
        zonotope_weights[0, word_num, :] = center = all_embeddings_for_position.mean(dim=0)  # before mean: (1 + NumSynonyms, E)   after: (E)
        distances_from_center = (center - all_embeddings_for_position)  # (distance to center, per dimension, for every element)

        N_word_plus_synonym = all_embeddings_for_position.size(0)
        assert (distances_from_center == 0).all().item() == (N_word_plus_synonym == 1), "Logic error in my reasoning"
        if (distances_from_center == 0).all() or (N_word_plus_synonym == 1):
            error_index += E
            continue
        # elif N_word_plus_synonym == 2:
        #     assert torch.allclose(distances_from_center[0], -distances_from_center[1], atol=1e-6), "Logic error in distances in my reasoning"
        #     zonotope_weights[error_index, word_num, :] = distances_from_center[0]
        #     error_index += E
        #     continue
        elif p > 10:
            sentence_embeddings_lower, _ = all_embeddings_for_position.min(dim=0)
            sentence_embeddings_upper, _ = all_embeddings_for_position.max(dim=0)
            zonotope_weights[0, word_num, :] = center = 0.5 * (sentence_embeddings_lower + sentence_embeddings_upper)
            coeffs = 0.5 * (sentence_embeddings_upper - sentence_embeddings_lower)
        elif use_solver:
            coeffs = get_coeffs_using_solver(distances_from_center, p)
        else:
            radiuses_per_word = get_norm(distances_from_center, p=p, dim=1)  # (1 + NumSynonyms)
            coeff = radiuses_per_word.max()
            coeffs = coeff.repeat(E)

        for embedding_dim in range(E):
            zonotope_weights[error_index, word_num, embedding_dim] = coeffs[embedding_dim]
            error_index += 1

        # if max_radius > 0:
        #     if p == 2:
        #         # volume *= math.pi**(E/2) / math.gamma(E/2 + 1) * (max_radius**E)
        #         log_volume += ((E/2) * math.log(math.pi)) - (loggamma(E/2 + 1)) + (E * math.log(max_radius))
        #     elif p == 1:
        #         # volume *= (2 * 1)**E / math.gamma(E + 1) * (max_radius**E)
        #         log_volume += (E * math.log(2)) - (loggamma(E + 1)) + (E * math.log(max_radius))

    # print(f"Log Volume: {log_volume}")
    return zonotope_weights, number_neighbors, num_enumerations, candidate_words_per_position


def get_coeffs_using_solver(distances: torch.Tensor, p: float) -> torch.Tensor:
    assert p == 1 or p == 2, "get_coeffs_using_solver: P must be 1 or 2"
    N, E = distances.shape

    # p-norm(x1/c1, x2/c2, ...) <= 1 with all c > 0

    ## CREATE MODEL
    model = Model("SynonymAttackCoeffsModel")
    model.setParam('OutputFlag', 0)

    ## CREATE VARIABLES
    inv_coeff_vars = []
    for i in range(E):
        inv_coeff_var = model.addVar(vtype=GRB.CONTINUOUS, name=f"E{i}", lb=1e-6, ub=GRB.INFINITY)
        inv_coeff_vars.append(inv_coeff_var)

    ## SETUP NORM CONSTRAINTS
    # We setup the lower bound to 0 instead of -1, because this will
    # allow us to model the 1-norm constraint without the problem of dealing
    # with setting up absolute value constraints in Gurobi
    for i in range(N):
        # Eps for this datapoint must respect 1-norm
        if p == 1:
            p_constraint = quicksum([inv_coeff_vars[k] * abs(distances[i][k].item()) for k in range(E)])
        else:  # p = 2
            raise NotImplementedError("NOT DONE")
            # p_constraint = quicksum([eps_vars[i, k] * eps_vars[i, k] for k in range(E)])

        model.addConstr(p_constraint <= 1.0, name=f"EpsConstraint{i}")

        # Eps for this datapoint must make it possible to obtain the point
        # (if we forgot to add this constraint, the solver could just output eps=0)
        # for k in range(E):
        #     model.addConstr(coeff_vars[k] * eps_vars[i, k], GRB.EQUAL, abs(distances[i][k].item()), name=f"DataPoint[{i}][{k}]")

    ## SETUP THE OBJECTIVE
    obj = quicksum([var * var for var in inv_coeff_vars])
    model.setObjective(obj, GRB.MAXIMIZE)

    ## SOLVE
    model.setParam(GRB.Param.TimeLimit, 5)
    model.optimize()

    ## GET RESULTS
    if model.status == GRB.Status.OPTIMAL:
        model_result = torch.tensor(model.getAttr('x', inv_coeff_vars), device=distances.device)
        return 1 / model_result
    elif model.status == GRB.Status.TIME_LIMIT:
        if model.getAttr(GRB.Attr.SolCount) == 0:
            raise Exception("GUROBI: Optimization stopped due to time limit: no solution found.")
        else:
            model_result = torch.tensor(model.getAttr('x', inv_coeff_vars), device=distances.device)
            return 1 / model_result
    else:
        raise Exception("GUROBI: Couldn't find optimal solution! Model status code: %s" % model.status)












        # Calculate Al, Bl, Cl by sampling and linear programming.
        bndX = np.array([lx, lx, ux, ux])
        bndY = np.array([ly, uy, ly, uy])
        X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])  # -4 because we want to have the corners
        Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])  # -4 because we want to have the corners

        model = Model()


        Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Al')
        Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Bl')
        Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Cl')

        model.addConstrs((Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], name) \
                          for i in range(n_samples)), name='ctr')

        obj = LinExpr()
        obj = np.sum(f(X, Y, name)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.Status.OPTIMAL:
            Al, Bl, Cl = model.getAttr('x', model.getVars())
            delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, name)
            Cl += delta
            return Al, Bl, Cl
        else:
            return None, None, None





    return coeffs

