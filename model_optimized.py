import torch
import torch.nn.functional as F
import numpy as np
import warnings
from utils import to_dataset, to_dataset_ignore_na

# As opposed to torch.double
torch.set_default_dtype(torch.float32)

def _configure_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        is_amd = "AMD" in props.name or hasattr(torch.version, 'hip')
        
        if not is_amd:
            # NVIDIA-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            # AMD/ROCm — cudnn flags are no-ops, skip them
            pass
        return device, is_amd
    return torch.device("cpu"), False

device, IS_AMD = _configure_device()
torch.set_default_device(device)

class FeatureDependentMarkovChain():
    def __init__(self, num_states, mask=None, lam_frob=0.1, W_lap_states=None,
                 W_lap_features=None, lam_col_norm=0.0, eps=1e-6, n_iter=50,
                 batch_size=None, mini_batch_size=32):
        self.n = num_states
        self.n_iter = n_iter
        self.lam = lam_frob
        self.W_lap_states = W_lap_states
        self.W_lap_features = W_lap_features
        self.lam_col_norm = lam_col_norm
        self.eps = eps
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        if mask is None:
            self.mask = np.ones((self.n, self.n))
        else:
            assert mask.shape == (self.n, self.n)
            self.mask = mask
        self.nonzero = [np.where(self.mask[i])[0] for i in range(self.n)]
        self.zero = [np.where(1 - self.mask[i])[0] for i in range(self.n)]
        self.sizes = [len(n) for n in self.nonzero]
        for s in self.sizes:
            assert s > 0, "Mask must have at least one state > 0"

        # Pre-cache Laplacian edge indices as tensors so they aren't recomputed each step
        self._lap_state_edges = None
        self._lap_feature_edges = None
        self._lap_state_weights = None
        self._lap_feature_weights = None

    # ------------------------------------------------------------------
    # Internal: cache Laplacian edge tensors once per fit() call
    # ------------------------------------------------------------------
    def _cache_laplacian_edges(self):
        if self.W_lap_states is not None:
            rows, cols = self.W_lap_states.nonzero()
            self._lap_state_edges = (
                torch.from_numpy(rows).to(device),
                torch.from_numpy(cols).to(device),
            )
            self._lap_state_weights = torch.from_numpy(
                self.W_lap_states.data).to(device)
        if self.W_lap_features is not None:
            rows, cols = self.W_lap_features.nonzero()
            self._lap_feature_edges = (
                torch.from_numpy(rows).to(device),
                torch.from_numpy(cols).to(device),
            )
            self._lap_feature_weights = torch.from_numpy(
                self.W_lap_features.data).to(device)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_batched_data(self, states, features, lengths, use_predictions=False):
        sequence_indices = []
        start_idx = 0
        for length in lengths:
            if length > 1:
                sequence_indices.append((start_idx, start_idx + length))
            start_idx += length

        batch_size = len(sequence_indices) if self.batch_size is None else min(
            self.batch_size, len(sequence_indices))

        batched_data = []
        for batch_start in range(0, len(sequence_indices), batch_size):
            batch_sequences = sequence_indices[batch_start:batch_start + batch_size]

            X = {i: [] for i in range(self.n)}
            Y = {i: [] for i in range(self.n)}
            weights = {i: [] for i in range(self.n)}

            for si, ei in batch_sequences:
                s = states[si:ei]
                f = features[si:ei]

                if use_predictions and hasattr(self, 'As'):
                    Ps = self.predict(f[:-1])
                    l = to_dataset(list(Ps), s, f)
                else:
                    l = to_dataset_ignore_na(s, f, self.n)

                for feat, w, state, next_state in l:
                    zero = self.zero[state]
                    if np.any(next_state[zero] > 0):
                        warnings.warn(
                            f"Transition from {state} to {next_state} impossible "
                            "according to mask. Ignoring transition.")
                        continue
                    if np.any(np.isnan(feat)):
                        continue
                    X[state].append(feat)
                    Y[state].append(next_state)
                    weights[state].append(w)

            batched_data.append((X, Y, weights))

        return batched_data

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------
    def fit(self, states, features, lengths, verbose=False, warm_start=False, **kwargs):
        N, m = features.shape
        self.models = {}
        prev_loss = float("inf")
        self._cache_laplacian_edges()

        for k in range(self.n_iter):
            use_predictions = k > 0 and hasattr(self, 'As')
            batched_data = self._prepare_batched_data(
                states, features, lengths, use_predictions)

            X_all = {i: [] for i in range(self.n)}
            Y_all = {i: [] for i in range(self.n)}
            weights_all = {i: [] for i in range(self.n)}

            for X_batch, Y_batch, weights_batch in batched_data:
                for i in range(self.n):
                    X_all[i].extend(X_batch[i])
                    Y_all[i].extend(Y_batch[i])
                    weights_all[i].extend(weights_batch[i])

            ws, Xs, Ys = [], [], []
            for i in range(self.n):
                noutputs = self.sizes[i]
                if len(weights_all[i]) == 0:
                    warnings.warn(
                        f"No pairs found starting from state {i}. "
                        "Results from this state may be inaccurate.")
                    weightsi = np.ones(1)
                    Xi = np.zeros((1, m))
                    Yi = np.zeros((1, noutputs))
                    Yi[0, :] = 1 / noutputs
                else:
                    weightsi = np.array(weights_all[i])
                    Xi = np.array(X_all[i])
                    Yi = np.array(Y_all[i])
                ws.append(weightsi)
                Xs.append(Xi)
                Ys.append(Yi[:, self.nonzero[i]])

            if self.lam_col_norm == 0.0:
                self.As, self.bs, loss = self._logistic_regression_batched(
                    ws, Xs, Ys, self.lam, warm_start=warm_start,
                    W_lap_states=self.W_lap_states,
                    W_lap_features=self.W_lap_features, **kwargs)
            else:
                self.As, self.bs, loss = self._logistic_regression_column_norm_batched(
                    ws, Xs, Ys, self.lam, warm_start=warm_start,
                    W_lap_states=self.W_lap_states,
                    W_lap_features=self.W_lap_features,
                    lam_col_norm=self.lam_col_norm, **kwargs)

            if k > 0:
                if verbose:
                    print("%03d | %8.4e" % (k, -loss))
                if loss <= prev_loss and 1 - loss / prev_loss <= self.eps:
                    break
                prev_loss = loss

    # ------------------------------------------------------------------
    # Logistic regression (full-batch or mini-batch)
    # ------------------------------------------------------------------
    def _logistic_regression_batched(self, ws, Xs, Ys, lam, warm_start=False,
                                     W_lap_states=None, W_lap_features=None, **kwargs):

        # Use float32 on NVIDIA (big throughput gain vs float64)
        # Keep float64 on AMD until you've verified bfloat16 support on your card
        precision = torch.float32 if not IS_AMD else torch.float64

        m = Xs[0].shape[1]

        # Warm start path — cast when loading from numpy
        if warm_start and hasattr(self, "As") and hasattr(self, "bs"):
            As = [torch.from_numpy(A.copy()).to(device, dtype=precision).requires_grad_(True)
                for A in self.As]
            bs = [torch.from_numpy(b.copy()).to(device, dtype=precision).requires_grad_(True)
                for b in self.bs]
        # Cold start path — zeros inherit default dtype, so force it explicitly
        else:
            As = [torch.zeros(m, Ys[i].shape[1], dtype=precision,
                            device=device, requires_grad=True) for i in range(self.n)]
            bs = [torch.zeros(Ys[i].shape[1], dtype=precision,
                            device=device, requires_grad=True) for i in range(self.n)]

        # Upload all data to GPU once — pin_memory speeds the host→device copy
        ws_tensor = [torch.from_numpy(w).to(device, dtype=precision, non_blocking=True) for w in ws]
        Xs_tensor = [torch.from_numpy(X).to(device, dtype=precision, non_blocking=True) for X in Xs]
        Ys_tensor = [torch.from_numpy(Y).to(device, dtype=precision, non_blocking=True) for Y in Ys]
        total_weight = sum(w.sum().item() for w in ws_tensor)

        if self.mini_batch_size is None:
            return self._full_batch_training(
                As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                lam, W_lap_states, W_lap_features, total_weight)
        else:
            return self._mini_batch_training(
                As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                lam, W_lap_states, W_lap_features, total_weight)

    def _full_batch_training(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                             lam, W_lap_states, W_lap_features, total_weight):
        opt = torch.optim.LBFGS(As + bs, max_iter=50, tolerance_grad=1e-8,
                                 line_search_fn='strong_wolfe')

        def loss():
            opt.zero_grad()
            l = torch.tensor(0.0, device=device)
            for i in range(self.n):
                if len(ws_tensor[i]) == 0:
                    continue
                pred = F.log_softmax(Xs_tensor[i] @ As[i] + bs[i], dim=1)
                l = l + (F.kl_div(pred, Ys_tensor[i], reduction='none')
                         .sum(dim=1) * ws_tensor[i]).sum() / total_weight
                l = l + lam * As[i].pow(2).sum()

            if W_lap_states is not None or W_lap_features is not None:
                l = l + self._compute_laplacian_reg(As, bs,
                                                    W_lap_states, W_lap_features)
            l.backward()
            return l

        opt.step(loss)

        A_numpy = [A.detach().cpu().numpy() for A in As]
        b_numpy = [b.detach().cpu().numpy() for b in bs]
        return A_numpy, b_numpy, loss().item()

    def _mini_batch_training(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                             lam, W_lap_states, W_lap_features, total_weight):
        """
        Key GPU optimizations vs. original:
          • opt.zero_grad() / opt.step() called once per epoch (not per state)
          • torch.randperm pre-generated for all states at epoch start
          • Laplacian reg computed once per epoch, not per mini-batch
        """
        opt = torch.optim.Adam(As + bs, lr=0.01)

        # Pre-shuffle indices once (will be re-shuffled each epoch)
        n_samples = [len(ws_tensor[i]) for i in range(self.n)]
        n_epochs = 100

        for epoch in range(n_epochs):
            # Generate all random permutations at once (one CUDA call per state)
            indices = [torch.randperm(n_samples[i], device=device)
                       for i in range(self.n)]

            # Compute Laplacian reg once per epoch (not once per mini-batch)
            lap_reg = torch.tensor(0.0, device=device)
            if W_lap_states is not None or W_lap_features is not None:
                with torch.no_grad():
                    lap_reg = self._compute_laplacian_reg(
                        As, bs, W_lap_states, W_lap_features).detach()

            # Find the number of mini-batch steps (driven by largest state's data)
            max_steps = max(
                (n_samples[i] + self.mini_batch_size - 1) // self.mini_batch_size
                for i in range(self.n) if n_samples[i] > 0)

            epoch_loss = 0.0
            for step in range(max_steps):
                opt.zero_grad()
                total_loss = torch.tensor(0.0, device=device)

                for i in range(self.n):
                    if n_samples[i] == 0:
                        continue
                    bs_start = step * self.mini_batch_size
                    if bs_start >= n_samples[i]:
                        # Cycle through data if this state runs out first
                        bs_start = bs_start % n_samples[i]
                    bs_end = min(bs_start + self.mini_batch_size, n_samples[i])
                    idx = indices[i][bs_start:bs_end]

                    pred = F.log_softmax(Xs_tensor[i][idx] @ As[i] + bs[i], dim=1)
                    batch_loss = (F.kl_div(pred, Ys_tensor[i][idx], reduction='none')
                                  .sum(dim=1) * ws_tensor[i][idx]).sum() / total_weight
                    total_loss = total_loss + batch_loss + lam * As[i].pow(2).sum()

                # Laplacian reg added once (already detached above — re-attach for grad)
                if W_lap_states is not None or W_lap_features is not None:
                    total_loss = total_loss + self._compute_laplacian_reg(
                        As, bs, W_lap_states, W_lap_features)

                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.item()

            if epoch % 20 == 0 and epoch > 0:
                pass  # Uncomment to debug: print(f"Epoch {epoch} loss: {epoch_loss:.6f}")

        final_loss = self._compute_full_loss(
            As, bs, ws_tensor, Xs_tensor, Ys_tensor,
            lam, W_lap_states, W_lap_features, total_weight)

        A_numpy = [A.detach().cpu().numpy() for A in As]
        b_numpy = [b.detach().cpu().numpy() for b in bs]
        return A_numpy, b_numpy, final_loss

    # ------------------------------------------------------------------
    # Laplacian regularization — uses pre-cached edge tensors
    # ------------------------------------------------------------------
    def _compute_laplacian_reg(self, As, bs, W_lap_states, W_lap_features):
        reg = torch.tensor(0.0, device=device)

        if W_lap_states is not None and self._lap_state_edges is not None:
            rows, cols = self._lap_state_edges
            m = As[0].shape[0]
            # Build full parameter tensors once
            A_full = torch.zeros(self.n, m, self.n, device=device)
            b_full = torch.zeros(self.n, self.n, device=device)
            for i in range(self.n):
                A_full[i, :, self.nonzero[i]] = As[i]
                b_full[i, self.nonzero[i]] = bs[i]

            A_diff = (A_full[rows] - A_full[cols]).pow(2).sum((1, 2))
            b_diff = (b_full[rows] - b_full[cols]).pow(2).sum(1)
            reg = reg + ((A_diff + b_diff) * self._lap_state_weights).sum()

        if W_lap_features is not None and self._lap_feature_edges is not None:
            rows, cols = self._lap_feature_edges
            m = As[0].shape[0]
            A_full = torch.zeros(self.n, m, self.n, device=device)
            for i in range(self.n):
                A_full[i, :, self.nonzero[i]] = As[i]

            A_diff = (A_full[:, rows] - A_full[:, cols]).pow(2).sum((0, 2))
            reg = reg + (A_diff * self._lap_feature_weights).sum()

        return reg

    # ------------------------------------------------------------------
    # Full-loss helper
    # ------------------------------------------------------------------
    def _compute_full_loss(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                           lam, W_lap_states, W_lap_features, total_weight):
        total_loss = torch.tensor(0.0, device=device)
        for i in range(self.n):
            if len(ws_tensor[i]) == 0:
                continue
            pred = F.log_softmax(Xs_tensor[i] @ As[i] + bs[i], dim=1)
            data_loss = (F.kl_div(pred, Ys_tensor[i], reduction='none')
                         .sum(dim=1) * ws_tensor[i]).sum() / total_weight
            total_loss = total_loss + data_loss + lam * As[i].pow(2).sum()

        if W_lap_states is not None or W_lap_features is not None:
            total_loss = total_loss + self._compute_laplacian_reg(
                As, bs, W_lap_states, W_lap_features)

        return total_loss.item()

    def _logistic_regression_column_norm_batched(self, ws, Xs, Ys, lam,
                                                 warm_start=False,
                                                 W_lap_states=None,
                                                 W_lap_features=None,
                                                 lam_col_norm=0.1):
        return self._logistic_regression_batched(
            ws, Xs, Ys, lam, warm_start, W_lap_states, W_lap_features)

    # ------------------------------------------------------------------
    # predict — fully on GPU, returns numpy for compatibility
    # ------------------------------------------------------------------
    def predict(self, features):
        """
        GPU-accelerated prediction.
        features: (T, m) numpy array  →  returns (T, n, n) numpy array
        """
        with torch.no_grad():
            # Upload once
            F_t = torch.from_numpy(features).to(device, non_blocking=True)  # (T, m)
            T = F_t.shape[0]
            P = torch.zeros(T, self.n, self.n, device=device)

            for i in range(self.n):
                A_i = torch.from_numpy(self.As[i]).to(device, non_blocking=True)
                b_i = torch.from_numpy(self.bs[i]).to(device, non_blocking=True)
                logits = F_t @ A_i + b_i          # (T, |nonzero[i]|)
                probs = F.softmax(logits, dim=1)   # (T, |nonzero[i]|)
                nz = torch.from_numpy(self.nonzero[i]).to(device)
                P[:, i, :].scatter_(1, nz.expand(T, -1), probs)

        return P.cpu().numpy()  # (T, n, n)

    # ------------------------------------------------------------------
    # score — GPU-accelerated, avoids redundant full predict() calls
    # ------------------------------------------------------------------
    def score(self, states, features, lengths, average=False):
        X = {i: [] for i in range(self.n)}
        Y = {i: [] for i in range(self.n)}
        idx = 0
        for length in lengths:
            if length <= 1:
                idx += length
                continue
            s = states[idx:idx + length]
            f = features[idx:idx + length]
            l = to_dataset_ignore_na(s, f, self.n)
            for feat, w, state, next_state in l:
                if np.any(next_state[self.zero[state]] > 0):
                    warnings.warn(
                        f"Transition from {state} to {next_state} impossible. "
                        "Ignoring.")
                    continue
                if np.any(np.isnan(feat)):
                    continue
                X[state].append(feat)
                Y[state].append(next_state)
            idx += length

        ll = 0.0
        ct = 0
        with torch.no_grad():
            for i in range(self.n):
                if len(X[i]) == 0:
                    continue
                ct += len(X[i])
                # Only compute the row for state i — avoids full n×n prediction
                Xi = torch.from_numpy(np.array(X[i])).to(device, non_blocking=True)
                A_i = torch.from_numpy(self.As[i]).to(device, non_blocking=True)
                b_i = torch.from_numpy(self.bs[i]).to(device, non_blocking=True)
                logits = Xi @ A_i + b_i
                log_probs_nz = F.log_softmax(logits, dim=1)  # (T, |nonzero[i]|)

                Yi = np.array(Y[i])[:, self.nonzero[i]]   # (T, |nonzero[i]|)
                Yi_t = torch.from_numpy(Yi).to(device, non_blocking=True)

                # Replace -inf contributions with 0 (matches original behaviour)
                lp = log_probs_nz.clamp(min=torch.finfo(torch.double).min)
                ll += (lp * Yi_t).sum().item()

        if average:
            ll /= ct
        return ll
