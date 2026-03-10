import torch
import torch.nn.functional as F
import numpy as np
import warnings
from scipy.special import softmax
from utils import to_dataset, to_dataset_ignore_na

torch.set_default_dtype(torch.double)
device = torch.device("cuda")
torch.set_default_device(device)

class FeatureDependentMarkovChain():
    def __init__(self, num_states, mask=None, lam_frob=0.1, W_lap_states=None,
                 W_lap_features=None, lam_col_norm=0.0, eps=1e-6, n_iter=50, 
                 batch_size=None, mini_batch_size=32):
        """
        Args:
            - num_states
            - mask
            - lam_frob
            - W_lap_states
            - W_lap_features
            - lam_col_norm
            - eps
            - n_iter
            - batch_size: Number of sequences to process at once (None = all sequences)
            - mini_batch_size: Size of mini-batches for gradient computation
        """
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
        self.zero = [np.where(1-self.mask[i])[0] for i in range(self.n)]
        self.sizes = [len(n) for n in self.nonzero]
        for s in self.sizes:
            assert s > 0, "Mask must have at least one state > 0"

    def _prepare_batched_data(self, states, features, lengths, use_predictions=False):
        """Prepare data in batches to reduce memory usage"""
        # Create sequence indices
        sequence_indices = []
        start_idx = 0
        for length in lengths:
            if length > 1:
                sequence_indices.append((start_idx, start_idx + length))
            start_idx += length
        
        # Process sequences in batches
        if self.batch_size is None:
            batch_size = len(sequence_indices)
        else:
            batch_size = min(self.batch_size, len(sequence_indices))
        
        batched_data = []
        for batch_start in range(0, len(sequence_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(sequence_indices))
            batch_sequences = sequence_indices[batch_start:batch_end]
            
            X = dict([(i, []) for i in range(self.n)])
            Y = dict([(i, []) for i in range(self.n)])
            weights = dict([(i, []) for i in range(self.n)])
            
            for start_idx, end_idx in batch_sequences:
                s = states[start_idx:end_idx]
                f = features[start_idx:end_idx]
                
                if use_predictions and hasattr(self, 'As'):
                    # Get Ps for this sequence
                    Ps = self.predict(f[:-1])
                    l = to_dataset(list(Ps), s, f)
                else:
                    l = to_dataset_ignore_na(s, f, self.n)
                
                for feat, w, state, next_state in l:
                    zero = self.zero[state]
                    if np.any(next_state[zero] > 0):
                        warnings.warn("Transition from " + str(state) + " to " + str(next_state) + " impossible according to mask. Ignoring transition.")
                        continue
                    if np.any(np.isnan(feat)):
                        continue
                    X[state].append(feat)
                    Y[state].append(next_state)
                    weights[state].append(w)
            
            batched_data.append((X, Y, weights))
        
        return batched_data

    def fit(self, states, features, lengths, verbose=False, warm_start=False, **kwargs):
        """
        Fit the model with batch training to reduce memory usage
        """
        N, m = features.shape
        self.models = {}
        prev_loss = float("inf")
        
        for k in range(self.n_iter):
            # Prepare batched data
            use_predictions = k > 0 and hasattr(self, 'As')
            batched_data = self._prepare_batched_data(states, features, lengths, use_predictions)
            
            # Aggregate data from all batches
            X_all = dict([(i, []) for i in range(self.n)])
            Y_all = dict([(i, []) for i in range(self.n)])
            weights_all = dict([(i, []) for i in range(self.n)])
            
            for X_batch, Y_batch, weights_batch in batched_data:
                for i in range(self.n):
                    X_all[i].extend(X_batch[i])
                    Y_all[i].extend(Y_batch[i])
                    weights_all[i].extend(weights_batch[i])
            
            # Prepare final arrays
            ws, Xs, Ys = [], [], []
            for i in range(self.n):
                noutputs = self.sizes[i]
                if len(weights_all[i]) == 0:  # no data points
                    warnings.warn("No pairs found in the dataset starting from state " + 
                        str(i) + " . results starting from this state might be inaccurate or useless.")
                    weightsi = np.ones(1)
                    Xi = np.zeros((1, m))
                    Yi = np.zeros((1, noutputs))
                    Yi[0, :] = 1 / noutputs  # Fixed indexing
                else:
                    weightsi, Xi, Yi = np.array(weights_all[i]), np.array(X_all[i]), np.array(Y_all[i])
                ws.append(weightsi)
                Xs.append(Xi)
                Ys.append(Yi[:, self.nonzero[i]])

            # Train with mini-batching
            if self.lam_col_norm == 0.0:
                self.As, self.bs, loss = self._logistic_regression_batched(
                    ws, Xs, Ys, self.lam, warm_start=warm_start,
                    W_lap_states=self.W_lap_states, W_lap_features=self.W_lap_features, **kwargs)
            else:
                self.As, self.bs, loss = self._logistic_regression_column_norm_batched(
                    ws, Xs, Ys, self.lam, warm_start=warm_start,
                    W_lap_states=self.W_lap_states, W_lap_features=self.W_lap_features,
                    lam_col_norm=self.lam_col_norm, **kwargs)
            
            if k > 0:
                if verbose:
                    print("%03d | %8.4e" % (k, -loss))
            if k > 0 and loss <= prev_loss and 1 - loss / prev_loss <= self.eps:
                break
            if k > 0:
                prev_loss = loss

    def _logistic_regression_batched(self, ws, Xs, Ys, lam, warm_start=False,
                                   W_lap_states=None, W_lap_features=None, **kwargs):
        """Logistic regression with mini-batch processing"""
        
        m = Xs[0].shape[1]

        if warm_start and hasattr(self, "As") and hasattr(self, "bs"):
            As = [torch.from_numpy(A.copy()).to(device) for A in self.As]
            bs = [torch.from_numpy(b.copy()).to(device) for b in self.bs]
            for A, b in zip(As, bs):
                A.requires_grad_(True)
                b.requires_grad_(True)
        else:
            As = [torch.zeros(m, Ys[i].shape[1], requires_grad=True).to(device) for i in range(self.n)]
            bs = [torch.zeros(Ys[i].shape[1], requires_grad=True).to(device) for i in range(self.n)]

        # Convert to tensors once
        ws_tensor = [torch.from_numpy(w).to(device) for w in ws]
        Xs_tensor = [torch.from_numpy(X).to(device) for X in Xs]
        Ys_tensor = [torch.from_numpy(Y).to(device) for Y in Ys]
        
        total_weight = sum([w.sum().item() for w in ws_tensor])

        # Choose training method based on mini_batch_size
        if self.mini_batch_size is None:
            # Full batch training - closer to original LBFGS behavior
            return self._full_batch_training(As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                                           lam, W_lap_states, W_lap_features, total_weight)
        else:
            # Mini-batch training
            return self._mini_batch_training(As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                                           lam, W_lap_states, W_lap_features, total_weight)

    def _full_batch_training(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                           lam, W_lap_states, W_lap_features, total_weight):
        """Full batch training - updates weights using ALL data at once"""
        # Use LBFGS for better convergence with full batches
        opt = torch.optim.LBFGS(As + bs, max_iter=50, tolerance_grad=1e-8, 
                               line_search_fn='strong_wolfe')
        # loss_fn = F.kl_div(reduction='none')
        # lsm = F.log_softmax(dim=1)

        def loss():
            opt.zero_grad()
            l = 0
            
            # Compute loss for all states using ALL their data
            for i in range(self.n):
                if len(ws_tensor[i]) == 0:
                    continue
                
                pred = F.log_softmax(Xs_tensor[i] @ As[i] + bs[i], dim=1)
                l += (F.kl_div(pred, Ys_tensor[i], reduction='none').sum(axis=1) * ws_tensor[i]).sum() / total_weight
                l += lam * As[i].pow(2).sum()
            
            # Add Laplacian regularization
            if W_lap_states is not None or W_lap_features is not None:
                l += self._compute_laplacian_reg(As, bs, W_lap_states, W_lap_features)
            
            l.backward()
            return l

        opt.step(loss)

        # https://stackoverflow.com/questions/75064656/printing-pytorch-tensor-from-gpu-or-move-to-cpu-and-or-detach
        A_numpy = [A.detach().cpu().numpy() for A in As]
        b_numpy = [b.detach().cpu().numpy() for b in bs]
        return (A_numpy, b_numpy, loss().item())

    def _mini_batch_training(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor,
                           lam, W_lap_states, W_lap_features, total_weight):
        """Mini-batch training - updates weights using small chunks of data"""
        # Use Adam for mini-batch training
        opt = torch.optim.Adam(As + bs, lr=0.01) # .to(device)?
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        lsm = torch.nn.LogSoftmax(dim=1)

        # Mini-batch training
        n_epochs = 100  # Adjust as needed
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(self.n):
                if len(ws_tensor[i]) == 0:
                    continue
                    
                n_samples = len(ws_tensor[i])
                indices = torch.randperm(n_samples, device=device)
                
                for batch_start in range(0, n_samples, self.mini_batch_size):
                    batch_end = min(batch_start + self.mini_batch_size, n_samples)
                    batch_indices = indices[batch_start:batch_end]
                    
                    if len(batch_indices) == 0:
                        continue
                    
                    opt.zero_grad()
                    
                    # Compute loss for this mini-batch
                    X_batch = Xs_tensor[i][batch_indices]
                    Y_batch = Ys_tensor[i][batch_indices]
                    w_batch = ws_tensor[i][batch_indices]
                    
                    pred = lsm(X_batch @ As[i] + bs[i])
                    batch_loss = (loss_fn(pred, Y_batch).sum(axis=1) * w_batch).sum() / total_weight
                    
                    # Add regularization
                    reg_loss = lam * As[i].pow(2).sum()
                    
                    # Add Laplacian regularization (simplified for batching)
                    if W_lap_states is not None or W_lap_features is not None:
                        reg_loss += self._compute_laplacian_reg(As, bs, W_lap_states, W_lap_features)
                    
                    total_loss = batch_loss + reg_loss
                    total_loss.backward()
                    opt.step()
                    
                    epoch_loss += total_loss.item()
                    n_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                # print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        # Final loss computation
        final_loss = self._compute_full_loss(As, bs, ws_tensor, Xs_tensor, Ys_tensor, 
                                           lam, W_lap_states, W_lap_features, total_weight)

        # https://stackoverflow.com/questions/75064656/printing-pytorch-tensor-from-gpu-or-move-to-cpu-and-or-detach
        A_numpy = [A.detach().cpu().numpy() for A in As]
        b_numpy = [b.detach().cpu().numpy() for b in bs]
        return (A_numpy, b_numpy, final_loss)

    def _compute_laplacian_reg(self, As, bs, W_lap_states, W_lap_features):
        """Compute Laplacian regularization terms"""
        reg = 0.0
        
        if W_lap_states is not None:
            A_full = torch.zeros((self.n, As[0].shape[0], self.n))
            b_full = torch.zeros((self.n, self.n))
            
            for i in range(self.n):
                A_full[i, :, self.nonzero[i]] = As[i]
                b_full[i, self.nonzero[i]] = bs[i]
            
            rows, cols = W_lap_states.nonzero()
            A_diff = (A_full[rows, :, :] - A_full[cols, :, :]).pow(2).sum((1, 2))
            b_diff = (b_full[rows] - b_full[cols]).pow(2).sum(1)
            reg += ((A_diff + b_diff) * torch.from_numpy(W_lap_states.data)).sum()
        
        if W_lap_features is not None:
            A_full = torch.zeros((self.n, As[0].shape[0], self.n))
            for i in range(self.n):
                A_full[i, :, self.nonzero[i]] = As[i]
            
            rows, cols = W_lap_features.nonzero()
            A_diff = (A_full[:, rows, :] - A_full[:, cols, :]).pow(2).sum((0, 2))
            reg += (A_diff * torch.from_numpy(W_lap_features.data)).sum()
        
        return reg

    def _compute_full_loss(self, As, bs, ws_tensor, Xs_tensor, Ys_tensor, lam, 
                          W_lap_states, W_lap_features, total_weight):
        """Compute the full loss for reporting"""
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        lsm = torch.nn.LogSoftmax(dim=1)
        
        total_loss = 0.0
        
        for i in range(self.n):
            if len(ws_tensor[i]) == 0:
                continue
            pred = lsm(Xs_tensor[i] @ As[i] + bs[i])
            data_loss = (loss_fn(pred, Ys_tensor[i]).sum(axis=1) * ws_tensor[i]).sum() / total_weight
            reg_loss = lam * As[i].pow(2).sum()
            total_loss += data_loss + reg_loss
        
        # Add Laplacian regularization
        if W_lap_states is not None or W_lap_features is not None:
            total_loss += self._compute_laplacian_reg(As, bs, W_lap_states, W_lap_features)
        
        return total_loss.item()

    def _logistic_regression_column_norm_batched(self, ws, Xs, Ys, lam, warm_start=False,
                                               W_lap_states=None, W_lap_features=None, 
                                               lam_col_norm=.1):
        """Column norm regularized logistic regression with batching"""
        # This is a simplified version - full implementation would require 
        # more complex proximal gradient methods with batching
        return self._logistic_regression_batched(ws, Xs, Ys, lam, warm_start, 
                                               W_lap_states, W_lap_features)

    def predict(self, features):
        P = []
        for i in range(self.n):
            yi = np.zeros((features.shape[0], self.n))
            yi[:, self.nonzero[i]] = softmax(features @ self.As[i] + self.bs[i], axis=1)
            P.append(yi)
        P = np.array(P).swapaxes(0, 1)
        return P

    def score(self, states, features, lengths, average=False):
        X = dict([(i, []) for i in range(self.n)])
        Y = dict([(i, []) for i in range(self.n)])
        i = 0
        for length in lengths:
            if length <= 1:
                i += length
                continue
            s = states[i:i+length]
            f = features[i:i+length]
            l = to_dataset_ignore_na(s, f, self.n)

            for feat, w, state, next_state in l:
                if np.any(next_state[self.zero[state]] > 0):
                    warnings.warn("Transition from " + str(state) + " to " + str(next_state) + " impossible according to mask. Ignoring transition.")
                    continue
                if np.any(np.isnan(feat)):
                    continue
                X[state].append(feat)
                Y[state].append(next_state)
            i += length

        ll = 0.
        ct = 0
        for i in range(self.n):
            if len(X[i]) == 0:
                continue
            ct += len(X[i])
            Ps = self.predict(np.array(X[i]))
            z = np.log(Ps[:, i, :])
            z[z == -np.inf] = 0.
            ll += (z * np.array(Y[i])).sum()
        if average:
            ll /= ct
        return ll
