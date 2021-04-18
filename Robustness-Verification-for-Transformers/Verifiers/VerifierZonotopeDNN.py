import copy
import time

import torch

from Verifiers.Zonotope import Zonotope, cleanup_memory


# can only accept one example in each batch
class VerifierZonotopeDNN:
    def __init__(self, args, target, logger):
        self.args = args
        self.device = args.device
        self.target = target
        self.logger = logger
        self.res = args.res

        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps
        self.debug = args.debug
        self.verbose = args.debug or args.verbose
        self.method = args.method
        self.num_verify_iters = args.num_verify_iters
        self.max_eps = args.max_eps
        self.debug_pos = args.debug_pos
        self.perturbed_words = args.perturbed_words
        self.warmed = False

        self.hidden_act = args.hidden_act

    def run(self, data):
        examples = data
        i = 0
        for (images, labels) in examples:
            batch_size = images.shape[0]
            for j in range(batch_size):
                image = images[j].cuda()
                label = labels[j]

                start = time.time()
                lower_bound = self.verify(image, label)
                end = time.time()
                print("{}: {} {}".format(i, lower_bound, end - start))
                i += 1

                if i == 100:
                    return

    def verify_safety(self, label, embeddings, eps):
        cleanup_memory()
        errorType = OSError if self.debug else AssertionError

        # x = x.view(-1, self.layer_sizes[0])
        # for fc in self.fcs[:-1]:
        #     x = F.relu(fc(x))
        # return self.fcs[-1](x) # No ReLu on the last one

        try:
            with torch.no_grad():
                bounds = self._bound_input(embeddings, eps=eps)  # hard-coded yet

                for i, layer in enumerate(self.target.fcs[:-1]):
                    bounds = bounds.dense(layer)
                    bounds = bounds.relu()

                safety = self._bound_classifier(bounds, self.target.fcs[-1], label)

                return safety
        except errorType as err:  # for debug
            print("Warning: failed assertion", eps)
            print(err)
            return False

    def _bound_input(self, embeddings, eps) -> Zonotope:
        embeddings = embeddings.reshape(1, -1)
        bounds = Zonotope(self.args, p=self.p, eps=eps, perturbed_word_index=0, value=embeddings)
        return bounds

    def _bound_classifier(self, bounds: Zonotope, classifier, label) -> bool:
        # 1) They compute linear layer that computes the how higher class 0 is over class 1
        # 2) They multiply the bounds by that linear layer's matrix
        # 3) They concretize the bounds (e.g. they compute the actual values, instead of having error terms)
        # 4) They check if things are safe or not (e.g. if the lower bound of c0 - c1 > 0, then we're good)
        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        bounds = bounds.dense(classifier)

        l, u = bounds.concretize()

        if label == 0:
            safe = l[0][0] > 0
        else:
            safe = u[0][0] < 0

        return safe.item()

    def verify(self, embeddings, label):
        """ Verify the given example sentence """
        num_iters = self.num_verify_iters

        self.max_eps = 1.0
        l, r = 0, self.max_eps
        print("{:.5f} {:.5f}".format(l, r), end="")

        safe = self.verify_safety(label, embeddings, r)
        while safe:
            l = r
            r *= 2
            print("\r{:.5f} {:.5f}".format(l, r), end="")
            safe = self.verify_safety(label, embeddings, r)

        if l == 0:
            while not safe:
                r /= 2
                print("\r{:.5f} {:.5f}".format(l, r), end="")
                safe = self.verify_safety(label, embeddings, r)
            l, r = r, r * 2
            print("\r{:.5f} {:.5f}".format(l, r), end="")

        for j in range(num_iters):
            m = (l + r) / 2
            if self.verify_safety(label, embeddings, m):
                l = m
            else:
                r = m
            print("\r{:.5f} {:.5f}".format(l, r), end="")

        print()
        return l
