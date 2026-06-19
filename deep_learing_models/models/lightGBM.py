import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler


class LightGBMAnomalyDetector(nn.Module):
    """
    LightGBM-based Anomaly Detection with sliding window features
    """
    
    def __init__(self):
        super(LightGBMAnomalyDetector, self).__init__()
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def sliding_window_features(self, X, window_size=168):
        """
        Converts 1D time series to 2D lag feature matrix for tabular tree models.
        Uses fast NumPy strides to avoid memory copies.
        
        X: [T] or [B, T]
        Returns: [T, window_size] or [B, T, window_size]
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        is_batched = X.ndim == 3 or (X.ndim == 2 and X.shape[0] > 1 and X.shape[1] > window_size)
        
        if is_batched and X.ndim >= 2:
            # Handle batched input
            if X.ndim == 3:
                B, T, C = X.shape
                X = X.reshape(-1, C).flatten()
            else:
                B, T = X.shape
            
            windows_list = []
            for b in range(B):
                if X.ndim == 3:
                    windows = self._sliding_window_1d(X[b * T:(b + 1) * T], window_size)
                else:
                    windows = self._sliding_window_1d(X[b * T:(b + 1) * T], window_size)
                windows_list.append(windows)
            return np.vstack(windows_list)
        else:
            # Handle single sequence
            if X.ndim == 2:
                X = X.flatten()
            return self._sliding_window_1d(X, window_size)
    
    def _sliding_window_1d(self, X, window_size):
        """Create sliding window features from 1D array"""
        n_samples = len(X)
        if n_samples < window_size:
            return X.reshape(-1, 1)
        
        # Create windowed views using strides
        shape = (n_samples - window_size + 1, window_size)
        strides = (X.strides[0], X.strides[0])
        X_windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        
        # Pad the start with the first valid window row
        padding_len = window_size - 1
        padding = np.repeat(X_windows[0:1], padding_len, axis=0)
        
        return np.vstack([padding, X_windows])
    
    def fit(self, X_train, y_train, scale_pos_weight=1.0, n_estimators=256, 
            learning_rate=0.03, num_leaves=32, max_depth=4, subsample=0.5,
            colsample_bytree=0.8, min_data_in_leaf=40, window_size=168):
        """
        Fit LightGBM model on training data
        X_train: [T] or [B, T, C]
        y_train: [T] or [B, T] labels
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
        
        # Flatten if batched
        if X_train.ndim == 3:
            B, T, C = X_train.shape
            X_train = X_train.reshape(-1, C)
            y_train = y_train.reshape(-1)
        elif X_train.ndim == 2 and X_train.shape[0] > 1 and X_train.shape[1] > window_size:
            # Could be [B, T] - flatten to 1D
            X_train = X_train.flatten()
            if y_train.ndim == 2:
                y_train = y_train.flatten()
        
        # Convert to sliding window features
        X_windows = self.sliding_window_features(X_train, window_size=window_size)
        
        # Train LightGBM with specified hyperparameters
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            min_data_in_leaf=min_data_in_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X_windows, y_train)
        
        self.is_fitted = True
        self.window_size = window_size
    
    def predict_proba(self, X_test):
        """
        Compute anomaly probabilities for test data
        X_test: [T] or [B, T, C]
        Returns: [T] probabilities for anomaly class
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()
        
        is_batched = X_test.ndim == 3
        original_shape = X_test.shape if is_batched else (len(X_test),)
        
        # Flatten if batched
        if X_test.ndim == 3:
            B, T, C = X_test.shape
            X_test = X_test.reshape(-1, C)
        elif X_test.ndim == 2:
            X_test = X_test.flatten()
        
        # Convert to sliding window features
        X_windows = self.sliding_window_features(X_test, window_size=self.window_size)
        
        # Get probability predictions
        proba = self.model.predict_proba(X_windows)[:, 1]  # Probability of anomaly class
        
        if is_batched:
            proba = proba.reshape(original_shape[0], original_shape[1])
        
        return torch.from_numpy(proba).float()


class Model(nn.Module):
    """
    LightGBM Anomaly Detection Model for TSLib
    Supports all 5 time series analysis tasks
    Uses sliding window features for tabular tree model input
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Initialize LightGBM detector
        self.lgbm_detector = LightGBMAnomalyDetector()
        
        # LightGBM hyperparameters
        self.n_estimators = getattr(configs, 'lgbm_n_estimators', 256)
        self.learning_rate = getattr(configs, 'lgbm_learning_rate', 0.03)
        self.num_leaves = getattr(configs, 'lgbm_num_leaves', 32)
        self.max_depth = getattr(configs, 'lgbm_max_depth', 4)
        self.subsample = getattr(configs, 'lgbm_subsample', 0.5)
        self.colsample_bytree = getattr(configs, 'lgbm_colsample_bytree', 0.8)
        self.min_data_in_leaf = getattr(configs, 'lgbm_min_data_in_leaf', 40)
        self.window_size = getattr(configs, 'lgbm_window_size', 168)
        
        # Training parameters
        self.scale_pos_weight = 1.0
        
        # Anomaly threshold
        self.threshold = 0.5
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Fit LightGBM model on training data
        X_train: [B, T, C]
        y_train: [B, T]
        """
        # Calculate class weight
        n_anom = max(int(np.sum(y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train)), 1)
        y_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        self.scale_pos_weight = (len(y_np.flatten()) - n_anom) / n_anom
        
        self.lgbm_detector.fit(
            X_train,
            y_train,
            scale_pos_weight=self.scale_pos_weight,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_data_in_leaf=self.min_data_in_leaf,
            window_size=self.window_size
        )
        self.is_fitted = True
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass routing to task-specific methods
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc)
        return None
    
    def forecast(self, x_enc):
        """
        Forecast task - return anomaly probabilities as reconstruction
        x_enc: [B, T, C]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Compute anomaly probabilities
        scores = self.lgbm_detector.predict_proba(x_enc)  # [B, T]
        
        # Expand to output channels
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        B, T = scores.shape
        output = scores.unsqueeze(-1).expand(B, T, self.c_out)
        
        return output[:, -self.pred_len:, :]
    
    def imputation(self, x_enc):
        """
        Imputation task - return reconstruction based on anomaly probabilities
        x_enc: [B, T, C]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        scores = self.lgbm_detector.predict_proba(x_enc)  # [B, T]
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        B, T = scores.shape
        output = scores.unsqueeze(-1).expand(B, T, self.c_out)
        
        return output
    
    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task - return LightGBM-based anomaly probabilities
        x_enc: [B, T, C]
        Returns: [B, T, C] anomaly probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        scores = self.lgbm_detector.predict_proba(x_enc)  # [B, T]
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        B, T = scores.shape
        output = scores.unsqueeze(-1).expand(B, T, self.c_out)
        
        return output
    
    def classification(self, x_enc):
        """
        Classification task - aggregate probabilities for binary classification
        x_enc: [B, T, C]
        Returns: [B, num_class]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        scores = self.lgbm_detector.predict_proba(x_enc)  # [B, T]
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        B, T = scores.shape
        
        # Aggregate statistics for classification
        score_stats = torch.stack([
            scores.mean(dim=1),              # Mean anomaly probability
            scores.max(dim=1)[0],            # Max anomaly probability
            scores.std(dim=1),               # Std of anomaly probability
            (scores > self.threshold).float().mean(dim=1),  # Ratio of anomalies
        ], dim=1)
        
        return score_stats