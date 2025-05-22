import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --- Mô phỏng chuỗi hành vi robot đa chế độ ---
class MultiModeRobotSimulator:
    def __init__(self, dt=0.1, steps=50):
        self.dt = dt
        self.steps = steps
        self.speed = 1.0
        self.turn_rate = np.deg2rad(15)
        self.transition_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
        ])
        self.n_states = 3

    def simulate_sequence(self):
        x, y = 0.0, 0.0
        theta = 0.0
        state = np.random.choice(self.n_states)  # trạng thái khởi đầu ngẫu nhiên

        trajectory = []
        states = []
        for _ in range(self.steps):
            state = np.random.choice(self.n_states, p=self.transition_matrix[state])
            states.append(state)

            if state == 1:
                theta += self.turn_rate
            elif state == 2:
                theta -= self.turn_rate
            # đi thẳng thì giữ theta

            x += self.speed * self.dt * np.cos(theta)
            y += self.speed * self.dt * np.sin(theta)
            trajectory.append([x, y])
        return np.array(trajectory), np.array(states)

    def simulate_batch(self, n_sequences=300):
        trajectories = []
        true_modes = []
        for _ in range(n_sequences):
            traj, modes = self.simulate_sequence()
            trajectories.append(traj)
            true_modes.append(modes)
        return np.array(trajectories), np.array(true_modes)


# --- Transformer Encoder ---
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, model_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch, model_dim)
        x = x.permute(1, 0, 2)  # back to (batch, seq_len, model_dim)
        # Lấy embedding toàn chuỗi bằng pooling (mean)
        x = x.mean(dim=1)
        return x


# --- GMM class đơn giản (như trước) ---
class GaussianMixtureModel:
    def __init__(self, n_components=3, n_iter=100, tol=1e-6, verbose=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def _init_params(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)
        self.means_ = X[rng.choice(n_samples, self.n_components, replace=False)]
        self.weights_ = np.ones(self.n_components) / self.n_components
        cov_init = np.cov(X.T) + 1e-6 * np.eye(n_features)
        self.covariances_ = np.array([cov_init.copy() for _ in range(self.n_components)])

    def _estimate_gaussian_prob(self, X, mean, cov):
        n_features = X.shape[1]
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        norm_const = np.sqrt((2 * np.pi) ** n_features * cov_det)
        diff = X - mean
        exponent = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
        return np.exp(-0.5 * exponent) / norm_const

    def fit(self, X):
        n_samples, n_features = X.shape
        self._init_params(X)
        log_likelihood_old = None

        for i in range(self.n_iter):
            probs = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                probs[:, k] = self.weights_[k] * self._estimate_gaussian_prob(X, self.means_[k], self.covariances_[k])
            total_prob = probs.sum(axis=1)[:, np.newaxis]
            total_prob[total_prob == 0] = 1e-12
            resp = probs / total_prob

            log_likelihood = np.sum(np.log(total_prob))
            if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, log_likelihood={log_likelihood:.4f}")
                break
            log_likelihood_old = log_likelihood

            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_cov = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
                self.covariances_[k] = weighted_cov + 1e-6 * np.eye(n_features)

    def predict(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            probs[:, k] = self.weights_[k] * self._estimate_gaussian_prob(X, self.means_[k], self.covariances_[k])
        return np.argmax(probs, axis=1)


# --- Pipeline ---
if __name__ == "__main__":
    # 1. Tạo dữ liệu chuỗi đa chế độ
    simulator = MultiModeRobotSimulator()
    trajectories, true_modes = simulator.simulate_batch(n_sequences=300)
    print("Shape trajectories:", trajectories.shape)  # (300, steps, 2)

    # 2. Khởi tạo model Transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryTransformer(input_dim=2).to(device)
    model.eval()  # chỉ dùng để tạo embedding, không huấn luyện ở demo này

    # 3. Lấy embedding chuỗi
    with torch.no_grad():
        input_tensor = torch.tensor(trajectories, dtype=torch.float32).to(device)
        embeddings = model(input_tensor).cpu().numpy()
    print("Embedding shape:", embeddings.shape)  # (n_samples, model_dim)

    # 4. Áp dụng GMM clustering
    gmm = GaussianMixtureModel(n_components=3, verbose=True)
    gmm.fit(embeddings)
    pred_modes = gmm.predict(embeddings)

    # 5. Visualize kết quả embedding
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("True Modes (chế độ thực)")
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=true_modes[:, 0], cmap='tab10')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.title("GMM Predicted Modes")
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=pred_modes, cmap='tab10')
    plt.grid(True)
    plt.show()
