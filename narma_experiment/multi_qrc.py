import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
import perceval as pcvl
from merlin import QuantumLayer, ComputationSpace, LexGrouping

class HPT_QRC_Multi:
    def __init__(self, in_size=1, window=5, n_photons=3, n_reservoirs=3, n_virtual_nodes=3,
                 lex_out=10, ridge_alpha=1e-4, seed=42):
        self.in_size = in_size
        self.window = window
        self.n_input_modes = 1  # We flatten the input window, but typically feed it linearly. For multivariate we use PCA.
        self.n_memory_modes = 7
        self.n_modes = self.n_input_modes + self.n_memory_modes
        self.n_photons = n_photons
        self.n_reservoirs = n_reservoirs
        self.n_virtual_nodes = n_virtual_nodes
        self.lex_out = lex_out
        
        # For M-dimensional data over W window, total features = M * W
        self.total_enc = self.in_size * window
        
        # Adjust n_input_modes dynamically up to a maximum to fit W*M features
        if self.total_enc > 1:
            self.n_input_modes = min(5, self.total_enc) 
            self.n_memory_modes = max(3, 8 - self.n_input_modes)
            self.n_modes = self.n_input_modes + self.n_memory_modes
            
        self.input_state = [1] * n_photons + [0] * (self.n_modes - n_photons)
        
        self.reservoirs = self._build_reservoirs(seed)
        self.ridge = Ridge(alpha=ridge_alpha)
        
    def _build_temporal_circuit(self, n_modes, n_input_modes, n_steps, n_virtual_depth):
        circuit = pcvl.Circuit(n_modes)
        c = 0
        for t in range(n_steps):
            circuit.add(0, pcvl.GenericInterferometer(
                n_modes,
                lambda i, _t=t: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_i{i}"))
                                  // pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_o{i}"))),
                shape=pcvl.InterferometerShape.RECTANGLE,
            ))
            for m in range(n_input_modes):
                circuit.add(m, pcvl.PS(pcvl.P(f"input{c}")))
                c += 1

        for v in range(n_virtual_depth):
            circuit.add(0, pcvl.GenericInterferometer(
                n_modes,
                lambda i, _v=v: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_i{i}"))
                                  // pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_o{i}"))),
                shape=pcvl.InterferometerShape.RECTANGLE,
            ))
        return circuit, c

    def _build_reservoirs(self, base_seed):
        reservoirs = []
        for r_seed in range(self.n_reservoirs):
            for vd in range(1, self.n_virtual_nodes + 1):
                torch.manual_seed(base_seed + r_seed * 1000)
                # We need c encodings, where c = n_input_modes * window steps
                circ, n_enc = self._build_temporal_circuit(
                    self.n_modes, self.n_input_modes, self.window, vd
                )
                core = QuantumLayer(
                    input_size=n_enc,
                    circuit=circ,
                    input_state=self.input_state,
                    input_parameters=["input"],
                    trainable_parameters=["t"],
                    computation_space=ComputationSpace.UNBUNCHED,
                    dtype=torch.float32,
                )
                layer = nn.Sequential(core, LexGrouping(core.output_size, self.lex_out))
                layer.eval()
                reservoirs.append(layer)
                self.n_enc_actual = n_enc # store for padding padding
        return reservoirs
        
    def _quantum_features_batch(self, X_win):
        xt = torch.tensor(X_win, dtype=torch.float32)
        with torch.no_grad():
            feats = []
            for r in self.reservoirs:
                out = r(xt)
                if out.is_complex():
                    out = out.real
                feats.append(out.numpy())
            return np.concatenate(feats, axis=1)

    def _create_features(self, X):
        X_dim = self.in_size
        padded_X = np.vstack([np.zeros((self.window - 1, X_dim)), X.reshape(-1, X_dim)])
        X_win = []
        for t in range(self.window - 1, len(padded_X)):
            window_slice = padded_X[t - self.window + 1 : t + 1].flatten()
            if len(window_slice) < self.n_enc_actual:
                window_slice = np.pad(window_slice, (0, self.n_enc_actual - len(window_slice)))
            elif len(window_slice) > self.n_enc_actual:
                window_slice = window_slice[:self.n_enc_actual]
            X_win.append(window_slice)
            
        X_win = np.array(X_win)
        Q = self._quantum_features_batch(X_win)
        
        return np.hstack([Q, X_win])

    def fit(self, y_train, X_exog=None, discard_steps=100):
        if X_exog is not None:
            features = np.hstack([y_train, X_exog])
        else:
            features = y_train
            
        # Target is next step y
        X_in = features[:-1]
        y_target = y_train[1:]
            
        print(f"[HPT-QRC] Extracting quantum features for {len(X_in)} training steps...")
        Q_features = self._create_features(X_in)
        
        Q_features_fit = Q_features[discard_steps:, :]
        y_fit = y_target[discard_steps:]
        
        print("[HPT-QRC] Fitting Ridge readout...")
        self.ridge.fit(Q_features_fit, y_fit)
        
        self.last_X = features[-self.window:].copy()
        return self
        
    def predict(self, y_test, X_exog=None):
        if X_exog is not None:
            features = np.hstack([y_test, X_exog])
        else:
            features = y_test
            
        # For prediction, we stack the last known W steps before the test sequence
        concat_features = np.vstack([self.last_X, features])
        
        print(f"[HPT-QRC] Extracting quantum features for {len(features)} test steps...")
        Q_features = self._create_features(concat_features)
        
        # _create_features outputs one feature row for every step it can build a full window for.
        # Since we pre-pended exactly window-1 steps (from self.last_X), Q_features will have exactly len(features) rows.
        # Wait, self.last_X contains self.window length. So we actually pre-pend W steps.
        # But _create_features already handles its own internal padded_X which prepends zeros.
        # Wait, _create_features is designed to pad by zeros for the first W steps of the sequence passed to it.
        # If we pass concat_features, it will generate len(concat_features) features.
        # So we just want the last len(features) elements of Q_features!
        
        return self.ridge.predict(Q_features[-len(features):])
