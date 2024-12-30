import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

def computePairwiseDistances(X, perplexity, tolerance, maxIter):
    n = X.shape[0]
    P = np.zeros((n, n))
    log_perplexity = np.log(perplexity)
    
    pairwise_distances = np.sum(X**2, axis=1).reshape(-1, 1) + \
                         np.sum(X**2, axis=1) - 2 * X @ X.T
    
    for i in range(n):
        beta = 1.0 
        beta_min, beta_max = -np.inf, np.inf

        for _ in range(maxIter):
            exp_distances = np.exp(-pairwise_distances[i] * beta)
            exp_distances[i] = 0
            sum_exp = np.sum(exp_distances)
            H = np.log(sum_exp) + beta * np.sum(pairwise_distances[i] * exp_distances) / sum_exp
            
            H_diff = H - log_perplexity
            if np.abs(H_diff) < tolerance:
                break
            
            if H_diff > 0:
                beta_min = beta
                beta = (beta + beta_max) / 2 if beta_max != np.inf else beta * 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2 if beta_min != -np.inf else beta / 2
        
        P[i, :] = exp_distances / sum_exp
    
    P = (P + P.T) / (2 * n)
    return P


def computeLowDimAffinities(Y):
    n = Y.shape[0]
    pairwise_distances = np.sum(Y**2, axis=1).reshape(-1, 1) + \
                         np.sum(Y**2, axis=1) - 2 * Y @ Y.T
    inv_distances = 1 / (1 + pairwise_distances)
    np.fill_diagonal(inv_distances, 0) 
    Q = inv_distances / np.sum(inv_distances)
    return Q, inv_distances



def generateDataFromMixture(n, k, dim):
    means = [np.random.uniform(-10, 10, dim) for _ in range(k)]
    covariances = [np.random.rand(dim, dim) for _ in range(k)]
    covariances = [np.dot(cov, cov.T) + np.eye(dim) * 0.1 for cov in covariances]
    
    samples = np.random.multinomial(n, [1/k] * k)
    
    X = []
    labels = []
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        X.append(np.random.multivariate_normal(mean, cov, samples[i]))
        labels.append(np.full(samples[i], i))
    
    X = np.vstack(X)
    labels = np.concatenate(labels)
    
    return X, labels


def create_animation(frames, save_path="evolution.mp4"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def update(frame):
        P, Q, iteration = frame
        axes[0].clear()
        axes[1].clear()
        sns.heatmap(P, ax=axes[0], cbar=False)
        axes[0].set_title("Similarity Scores (High-Dimensional)")
        sns.heatmap(Q, ax=axes[1], cbar=False)
        axes[1].set_title(f"Low Dimension Similarities at Iteration {iteration}")
        fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    ani.save(save_path, writer="ffmpeg")
    plt.close(fig)



def t_SNE(X, dim, perplexity=5, iters=1001, lr=0.001, tolerance=1e-5, maxIter=1000, viz=False):

    n = X.shape[0]
    Y = np.random.normal(0, 0.0001, (n, dim))

    P = computePairwiseDistances(X, perplexity, tolerance, maxIter)
    P = np.maximum(P, 1e-12)

    if viz:
        frames = []

    for i in range(iters): 

        Q, inv_distances = computeLowDimAffinities(Y)
        Q = np.maximum(Q, 1e-12)

        PQ_diff = (P - Q)[:, :, None]
        Y_diff = Y[:, None, :] - Y[None, :, :]
        grad = 4 * np.sum(PQ_diff * inv_distances[:, :, None] * Y_diff, axis=1)

        Y -= lr * grad

        if i % 1 == 0:
            kl_div = np.sum(P * np.log(P / Q))
            print(f"Iteration {i}: KL Divergence = {kl_div:.4f}")

            if viz:
                frames.append((P.copy(), Q.copy(), i))

    if viz:
        create_animation(frames)

    return Y



if __name__ == "__main__":
    
    X, labels = generateDataFromMixture(100, 5, 10)
    print(labels)
    Y = t_SNE(X, 2, perplexity=10, iters=201, lr=10, tolerance=1e-5, maxIter=1000, viz=True)
    
    plt.scatter(Y[:, 0], Y[:, 1], c=labels)
    plt.show()