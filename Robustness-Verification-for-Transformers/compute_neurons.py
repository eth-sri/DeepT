def compute_neurons(N: int, E: int, H: int, A: int, num_blocks: int, our_softmax: bool):
    num_neurons = 0
    d_v = E
    d_k = E // A

    num_neurons += N * E  # Layer norm ()
    num_neurons += N * E  # Layer Normalization - "(N, E)"

    for j in range(num_blocks):
        for i in range(A):
            num_neurons += N * d_k  # Keys - "(N, d_k)"
            num_neurons += N * d_k  # Queries - "(N, d_k)"
            num_neurons += N * d_v  # Values - "(N, d_v)"
            num_neurons += N * N  # Attention Scores - "(N, N)"

            if our_softmax:
                num_neurons += N * N * N  # Softmax - Diffs - "(N, N, N)
                num_neurons += N * N * N  # Softmax - Diffs Exp - "(N, N, N)
                num_neurons += N * N  # Softmax - Diffs Exp Sum - "(N, N)"
                num_neurons += N * N  # Softmax - result = attention probs - "(N, N)"
            else: # Normal softmax
                num_neurons += N * N  # Softmax - Exps - "(N, N)"
                num_neurons += N * N  # Softmax - Exp Sums - "(N)"
                num_neurons += N * N  # Softmax - Ratio - "(N, N)"


            num_neurons += N * d_v  # Self-attention-output - "(N, d_v)"

        num_neurons += N * E  # Residual connection - "(N, E)"
        num_neurons += N * E  # Layer Norm - "(N, E)"
        num_neurons += N * H  # Projection 1 - "(N, E_2)"
        num_neurons += N * H  # Relu - "(N, E_2)"
        num_neurons += N * E  # Projection 2 - "(N, E)"
        num_neurons += N * E  # Residual connection - "(N, E)"
        num_neurons += N * E  # Layer Norm - "(N, E)"

    num_neurons += E  # Pooling - Dense,(E)
    num_neurons += E  # Pooling - Tanh,(E)
    num_neurons += 2  # Classifier,2
    return num_neurons


print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=3, our_softmax=True)=} neurons")
print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=3, our_softmax=False)=} neurons")
print()
print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=6, our_softmax=True)=} neurons")
print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=6, our_softmax=False)=} neurons")
print()
print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=12, our_softmax=True)=} neurons")
print(f"{compute_neurons(N=20, E=128, H=128, A=4, num_blocks=12, our_softmax=False)=} neurons")
