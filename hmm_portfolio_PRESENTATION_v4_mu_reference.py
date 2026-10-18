"""
HMM REGIME-SWITCHING PORTFOLIO OPTIMIZER (ETF)

DESCRIPTION:
    ETF portfolio optimization model based on a Hidden Markov Model (HMM) for
    market-regime detection. For each identified regime (Bull, Bear, Sideways)
    a separate (μ, Σ) pair is estimated. The portfolio is then optimized either
    by Joint Stochastic Programming (JSP) over simulated regime paths, or as a
    probability-weighted average of per-regime portfolios (weighting scheme
    configurable by the user).

REQUIREMENTS:
    pip install numpy pandas scipy scikit-learn hmmlearn matplotlib
               openpyxl yfinance cvxpy seaborn

USAGE:
    - Edit the "USER CONFIG" section at the bottom of the file
    - Enter your ETFs in the etf_tickers list
    - If you have a FactSet Excel file, set excel_file_path
    - Otherwise the data are downloaded automatically from Yahoo Finance
    - Run: python hmm_portfolio_optimizer.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from scipy.optimize import minimize, linear_sum_assignment, linprog
from scipy.stats import norm as scipy_norm
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sk_silhouette_score
from hmmlearn.hmm import GaussianHMM

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[WARN] cvxpy not installed. Falling back to scipy.optimize.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARN] yfinance not installed. Only Excel mode is available.")


# ══════════════════════════════════════════════════════════════════════════════
# MARCHENKO-PASTUR DENOISING — Layer post-Ledoit-Wolf
# ══════════════════════════════════════════════════════════════════════════════

def _mp_denoise_covariance(sigma_lw: np.ndarray,
                            returns_data: np.ndarray) -> tuple:

    n = sigma_lw.shape[0]
    t = len(returns_data)

    if t < 15:
        return sigma_lw, {'skipped': True, 'reason': f'n_obs={t}<15'}

    std_diag = np.sqrt(np.maximum(np.diag(sigma_lw), 1e-10))
    inv_std  = 1.0 / std_diag
    corr_lw  = sigma_lw * np.outer(inv_std, inv_std)
    np.fill_diagonal(corr_lw, 1.0)

    if returns_data.ndim == 2 and returns_data.shape[1] == n:
        rhos = []
        for i in range(n):
            col = returns_data[:, i]
            if len(col) > 2 and not np.isnan(col).any():
                ac = np.corrcoef(col[:-1], col[1:])[0, 1]
                if not np.isnan(ac):
                    rhos.append(abs(ac))
        rho_mean = min(0.5, np.mean(rhos) if rhos else 0.0)
    else:
        rho_mean = 0.0

    t_eff    = max(n + 1, int(t / (1 + 2 * rho_mean)))
    gamma_eff = n / t_eff
    lambda_plus = (1 + np.sqrt(gamma_eff)) ** 2

    eigvals, eigvecs = np.linalg.eigh(corr_lw)  # ascending order

    signal_mask = eigvals > lambda_plus
    n_signal_mp = int(np.sum(signal_mask))

    min_signal = max(1, int(np.ceil(n * 0.20)))
    n_signal   = max(n_signal_mp, min_signal)

    final_mask = np.zeros(n, dtype=bool)
    final_mask[np.argsort(eigvals)[-n_signal:]] = True

    n_noise    = int(np.sum(~final_mask))
    gamma_fill = (np.sum(eigvals) - np.sum(eigvals[final_mask])) / max(1, n_noise)
    gamma_fill = max(1e-6, gamma_fill)
    eigvals_clean = np.where(final_mask, eigvals, gamma_fill)

    corr_mp = (eigvecs * eigvals_clean) @ eigvecs.T
    corr_mp = (corr_mp + corr_mp.T) / 2.0
    d = np.sqrt(np.maximum(np.diag(corr_mp), 1e-10))
    corr_mp /= np.outer(d, d)
    np.fill_diagonal(corr_mp, 1.0)

    min_eig = np.linalg.eigvalsh(corr_mp).min()
    if min_eig < 1e-8:
        corr_mp += (abs(min_eig) + 1e-6) * np.eye(n)
        d2 = np.sqrt(np.maximum(np.diag(corr_mp), 1e-10))
        corr_mp /= np.outer(d2, d2)
        np.fill_diagonal(corr_mp, 1.0)

    sigma_mp = corr_mp * np.outer(std_diag, std_diag)

    JS_C_MP  = 30.0
    alpha_lw = JS_C_MP / (JS_C_MP + t)
    sigma_final = (1 - alpha_lw) * sigma_mp + alpha_lw * sigma_lw

    info = {
        'skipped':     False,
        'gamma_eff':   gamma_eff,
        'lambda_plus': lambda_plus,
        'n_signal':    n_signal,
        'n_noise':     n - n_signal,
        'rho_mean':    rho_mean,
        't_eff':       t_eff,
        'alpha_lw':    alpha_lw
    }
    return sigma_final, info

def _ledoit_wolf_cov(data: np.ndarray, preserve_variances: bool = True):
    """Ledoit-Wolf covariance; returns (cov, shrinkage intensity).

    scikit-learn's LedoitWolf shrinks toward mu*I (mu = average variance), which pulls EVERY
    variance toward the cross-sectional mean: with an intensity of 3% no asset can end up below
    sqrt(0.03) x the average volatility (~2.6% for a 15%-vol universe), so cash-like and
    short-duration ETFs get their risk overstated by 10-80x. With preserve_variances=True the
    returns are standardised first, LW shrinks the CORRELATION matrix, and the individual
    variances are restored afterwards (the diagonal stays exactly the sample variance).
    """
    data = np.asarray(data, dtype=float)
    if not preserve_variances:
        lw = LedoitWolf().fit(data)
        return lw.covariance_, float(lw.shrinkage_)
    sd = data.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    lw = LedoitWolf().fit(data / sd)
    return lw.covariance_ * np.outer(sd, sd), float(lw.shrinkage_)


def _sector_groups(ticker_names, sector_map, sector_constraints):
    """[(indices, lo, hi)] for every constrained sector that contains at least one ETF."""
    groups = []
    if sector_map and sector_constraints:
        for sector, (lo, hi) in sector_constraints.items():
            idx = [i for i, t in enumerate(ticker_names) if sector_map.get(t, '') == sector]
            if idx:
                groups.append((idx, lo, hi))
    return groups


def _slsqp_constraints(ticker_names, sector_map, sector_constraints):
    """SLSQP constraints: full investment + ALL sector bounds as one vector constraint (A w >= b)."""
    n = len(ticker_names)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones_like(w)}]
    groups = _sector_groups(ticker_names, sector_map, sector_constraints)
    if groups:
        A = np.zeros((2 * len(groups), n))
        b = np.zeros(2 * len(groups))
        for k, (idx, lo, hi) in enumerate(groups):
            A[2 * k, idx], b[2 * k] = 1.0, lo            #  sum(w_idx) >= lo
            A[2 * k + 1, idx], b[2 * k + 1] = -1.0, -hi  # -sum(w_idx) >= -hi
        cons.append({'type': 'ineq', 'fun': lambda w: A @ w - b, 'jac': lambda w: A})
    return cons


def _js_base_intensity(n_obs: int, n_assets: int) -> Tuple[float, str, bool]:
    """Base James-Stein intensity C: stronger shrinkage for short histories and many assets.

    Returns (C, description of the history tier, whether the N-aware floor was binding).
    """
    if n_obs < 260:
        c, tier = 80.0, "Short ETF series (~1 year)"
    elif n_obs < 520:
        c, tier = 50.0, "Medium ETF series (~2 years)"
    elif n_obs < 780:
        c, tier = 35.0, "Adequate ETF series (~3 years)"
    else:
        c, tier = 20.0, "Long ETF series (3+ years)"
    floor = max(1, n_assets - 2) * 2.0
    return max(c, floor), tier, c < floor


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:

    def __init__(self, excel_path: Optional[str] = None,
                 excel_sheet: str = "Sheet1",
                 excel_date_col: str = "Date"):
        self.excel_path = excel_path
        self.excel_sheet = excel_sheet
        self.excel_date_col = excel_date_col

    def load_from_excel(self, tickers: List[str], period_years: int = 5) -> pd.DataFrame:

        print(f"\n[INFO] Loading data from Excel: {self.excel_path}")
        try:
            df = pd.read_excel(self.excel_path, sheet_name=self.excel_sheet)

            if self.excel_date_col in df.columns:
                date_cols = [self.excel_date_col]
            else:
                date_cols = [c for c in df.columns if 'date' in str(c).lower() or 'data' in str(c).lower()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'Date'})
            else:
                df.columns.values[0] = 'Date'

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all')

            available = [c for c in tickers if c in df.columns]
            if len(available) < len(tickers):
                missing = [t for t in tickers if t not in df.columns]
                print(f"[WARN] Tickers not found in Excel (will be downloaded from Yahoo): {missing}")
                df_missing = self._download_yfinance(missing, df.index[0], df.index[-1])
                df = pd.concat([df[available], df_missing], axis=1)
            else:
                df = df[tickers]

            df = df.ffill(limit=5)
            returns = np.log(df / df.shift(1)).dropna()
            print(f"[OK] Excel data loaded: {len(returns)} observations, {len(returns.columns)} ETFs")
            print(f"     Period: {returns.index[0].date()} → {returns.index[-1].date()}")
            return returns

        except Exception as e:
            print(f"[ERROR] Unable to read Excel file: {e}")
            print("[INFO] Falling back to Yahoo Finance...")
            return self.load_from_yfinance(tickers, period_years=period_years)

    @staticmethod
    def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Close-price frame whose columns are labelled by the downloaded ticker.

        yfinance returns columns in alphabetical order, so labels must come from
        the download itself and never be re-assigned positionally.
        """
        if isinstance(raw.columns, pd.MultiIndex):
            return raw['Close']
        if len(tickers) == 1:
            return raw[['Close']].rename(columns={'Close': tickers[0]})
        return raw

    def _download_yfinance(self, tickers: List[str],
                           start: pd.Timestamp,
                           end: pd.Timestamp) -> pd.DataFrame:
        """Adjusted close PRICES (not returns) for tickers missing from the Excel file."""
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance unavailable and tickers not found in Excel.")
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        prices = self._extract_close(raw, tickers)
        return prices.dropna(how='all')

    def load_from_yfinance(self, tickers: List[str],
                           period_years: int = 5) -> pd.DataFrame:

        if not YFINANCE_AVAILABLE:
            raise RuntimeError("Please install yfinance: pip install yfinance")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=period_years * 365)

        print(f"\n[INFO] Downloading Yahoo Finance data ({period_years} years)...")
        print(f"       Tickers: {tickers}")

        raw = yf.download(tickers, start=start_date, end=end_date,
                          progress=False, auto_adjust=True)

        prices = self._extract_close(raw, tickers)
        prices = prices.dropna(how='all')

        for ticker in tickers:
            if ticker not in prices.columns:
                print(f"[WARN] {ticker} not found on Yahoo Finance — will be excluded")

        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
        # Exchanges close on different days: forward-fill short gaps so that a
        # single holiday does not delete that day and the next one for ALL ETFs.
        prices = prices.ffill(limit=5)

        first_valid = prices.apply(lambda col: col.first_valid_index())
        print("\n[DIAGNOSTIC] First available date for each ETF:")
        for t in available:
            n_days = prices[t].dropna().shape[0]
            fv = first_valid[t]
            fv_str = fv.date() if fv is not None else "N/A"
            print(f"  {t:12s}: first data point = {fv_str}, n={n_days} days")

        returns_raw = np.log(prices / prices.shift(1))
        returns = returns_raw.dropna(how='any')

        if len(returns) < len(returns_raw.dropna(how='all')):
            oldest = returns_raw.dropna(how='all').index[0].date()
            newest = returns.index[0].date() if len(returns) > 0 else "N/A"
            n_lost = (pd.Timestamp(newest) - pd.Timestamp(oldest)).days if newest != "N/A" else 0
            print(f"\n[INFO] ETF intersection: {newest} → {returns.index[-1].date() if len(returns) else 'N/A'}")
            print(f"       Days lost because of the most recent ETFs: ~{n_lost}d")
            print("       To use a longer history: remove ETFs with a short history from the portfolio.")

        print(f"[OK] Yahoo data loaded: {len(returns)} observations, {len(returns.columns)} ETFs")
        print(f"     Period: {returns.index[0].date()} → {returns.index[-1].date()}")
        return returns


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HIDDEN MARKOV MODEL — REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class MarketRegimeHMM:

    REGIME_LABELS = {0: "Sideways/Kangaroo", 1: "Bull", 2: "Bear"}

    def __init__(self, n_regimes: int = 3, n_iter: int = 150,
                 window_volatility: int = 10, random_state: int = 42,
                 regime_prob_floor: float = 0.05):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.window_volatility = window_volatility
        self.random_state = random_state
        self.regime_prob_floor = regime_prob_floor  # minimum floor per regime (avoids 97%+)
        self.model = None
        self.scaler = None
        self.regime_sequence = None
        self.regime_probs = None
        self.regime_mapping = {}
        self.is_fitted = False

    def _prepare_observations(self, benchmark_returns: pd.Series,
                               fit_scaler: bool = True) -> np.ndarray:

        ret = benchmark_returns.values
        vol = pd.Series(ret).rolling(self.window_volatility).std().bfill().ffill().values
        obs_raw = np.column_stack([ret, vol])

        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            obs_scaled = self.scaler.fit_transform(obs_raw)
        else:
            obs_scaled = self.scaler.transform(obs_raw)

        return obs_scaled

    def _select_n_regimes(self, obs: np.ndarray,
                          max_regimes: int = 5,
                          min_obs_per_regime: int = 40) -> int:

        n_obs = len(obs)
        k_max_data  = max(2, n_obs // min_obs_per_regime)
        k_max_final = min(max_regimes, k_max_data)

        print("\n[INFO] Data-adaptive BIC regime selection...")
        print(f"       n_obs={n_obs} | min_obs/regime={min_obs_per_regime} | "
              f"k_max_allowed={k_max_final}")

        if k_max_final < 2:
            print("       [WARN] Insufficient data for BIC — using k=2")
            return 2

        scores = {}
        for k in range(2, k_max_final + 1):
            try:
                m = GaussianHMM(n_components=k, covariance_type="full",
                                n_iter=self.n_iter, random_state=self.random_state,
                                min_covar=1e-3)
                m.fit(obs)
                ll    = m.score(obs)
                n_par = k*k + k*obs.shape[1] + k*obs.shape[1]*(obs.shape[1]+1)/2
                bic   = -2 * ll + n_par * np.log(n_obs)
                scores[k] = bic
                print(f"       k={k}: BIC={bic:.2f}, LogLik={ll:.2f} (~{n_obs//k} obs/regime)")
            except Exception as e:
                print(f"       k={k}: failed ({e})")

        if not scores:
            print("       [WARN] No model converged — using k=3")
            return 3

        best_k = min(scores, key=scores.get)
        print(f"[OK] Selected regimes: {best_k} (BIC={scores[best_k]:.2f})")
        return best_k

    def _map_regimes_economically(self) -> None:

        regime_stats = {}
        benchmark_arr = getattr(self, "benchmark_returns_aligned", None)
        for s in range(self.n_regimes):
            mask = self.regime_sequence == s
            if mask.sum() == 0:
                continue
            if benchmark_arr is not None and len(benchmark_arr) == len(self.regime_sequence):
                r = benchmark_arr[mask]
                if len(r) > 0:
                    mean_ret = float(np.mean(r))
                    vol = float(np.std(r))

                    max_dd = 0.0
                    in_block = False
                    block_returns = []
                    for t_idx, in_regime in enumerate(mask):
                        if in_regime:
                            block_returns.append(float(benchmark_arr[t_idx]))
                            in_block = True
                        else:
                            if in_block and len(block_returns) > 1:
                                cum_b = np.exp(np.cumsum(block_returns))
                                run_max_b = np.maximum.accumulate(cum_b)
                                dd_b = (cum_b - run_max_b) / np.maximum(run_max_b, 1e-12)
                                max_dd = min(max_dd, float(np.min(dd_b)))
                            block_returns = []
                            in_block = False
                    if in_block and len(block_returns) > 1:
                        cum_b = np.exp(np.cumsum(block_returns))
                        run_max_b = np.maximum.accumulate(cum_b)
                        dd_b = (cum_b - run_max_b) / np.maximum(run_max_b, 1e-12)
                        max_dd = min(max_dd, float(np.min(dd_b)))
                else:
                    mean_ret = float(self.model.means_[s][0])
                    vol = float(self.model.means_[s][1])
                    max_dd = 0.0
            else:
                mean_ret = float(self.model.means_[s][0])
                vol = float(self.model.means_[s][1])
                max_dd = 0.0
            time_share = float(mask.sum()) / max(len(self.regime_sequence), 1)
            regime_stats[s] = {
                'mean_return': mean_ret,
                'mean_vol': vol,
                'count': int(mask.sum()),
                'time_share': time_share,
                'max_drawdown': max_dd,
                'is_real_space': bool(benchmark_arr is not None and len(benchmark_arr) == len(self.regime_sequence))
            }

        states = list(regime_stats.keys())
        if len(states) < 2:
            self.regime_mapping = {states[0]: 0} if states else {}
            return

        mean_vals = np.array([regime_stats[s]['mean_return'] for s in states], dtype=float)
        vol_vals = np.array([regime_stats[s]['mean_vol'] for s in states], dtype=float)
        pers_vals = np.array([regime_stats[s]['time_share'] for s in states], dtype=float)
        dd_vals = np.array([regime_stats[s]['max_drawdown'] for s in states], dtype=float)

        def _z(v: np.ndarray) -> np.ndarray:
            std = float(np.std(v))
            if std < 1e-10:
                return np.zeros_like(v)
            return (v - np.mean(v)) / std

        z_mean = _z(mean_vals)
        z_vol = _z(vol_vals)
        z_pers = _z(pers_vals)
        z_dd = _z(dd_vals)

        bull_score = 0.50 * z_mean - 0.20 * z_vol + 0.15 * z_pers + 0.15 * z_dd
        bear_score = -0.45 * z_mean + 0.25 * z_vol + 0.20 * z_pers - 0.10 * z_dd

        bull_state = states[int(np.argmax(bull_score))]
        bear_state = states[int(np.argmax(bear_score))]
        if bull_state == bear_state:
            bull_state = states[int(np.argmax(mean_vals))]
            bear_state = states[int(np.argmin(mean_vals))]

        remaining = [s for s in states if s not in (bull_state, bear_state)]

        if self.n_regimes == 3:
            self.regime_mapping = {
                bull_state: 1,   # Bull
                bear_state: 2,   # Bear
                remaining[0] if remaining else bear_state: 0,   # Sideways
            }
        elif self.n_regimes == 2:
            self.regime_mapping = {
                bull_state: 1,   # Bull
                bear_state: 2,   # Bear
            }
        else:
            order = [states[i] for i in np.argsort(bull_score)]
            self.regime_mapping = {s: i for i, s in enumerate(order)}

        print("\n[INFO] Economic regime mapping:")
        for hmm_state, economic_state in self.regime_mapping.items():
            label = self.REGIME_LABELS.get(economic_state, f"Regime {economic_state}")
            stats = regime_stats.get(hmm_state, {})
            mean_zscored = stats.get('mean_return', 0)
            mean_vol_zscored = stats.get('mean_vol', 0)
            if stats.get('is_real_space', False):
                print(f"       HMM state {hmm_state} -> {label} | "
                      f"ret={mean_zscored*100:.3f}%/d ({mean_zscored*252*100:.1f}%/y) | "
                      f"vol={mean_vol_zscored*100:.4f}%/d | "
                      f"share={stats.get('time_share',0)*100:.1f}% | "
                      f"maxDD={stats.get('max_drawdown',0)*100:.1f}% | "
                      f"n={stats.get('count', 0)}")
            elif self.scaler is not None:
                try:
                    mean_real = mean_zscored * self.scaler.scale_[0] + self.scaler.mean_[0]
                    mean_vol_real = mean_vol_zscored * self.scaler.scale_[1] + self.scaler.mean_[1]
                    print(f"       HMM state {hmm_state} -> {label} | "
                          f"ret={mean_real*100:.3f}%/d ({mean_real*252*100:.1f}%/y) | "
                          f"vol={mean_vol_real*100:.4f}%/d | "
                          f"share={stats.get('time_share',0)*100:.1f}% | "
                          f"maxDD={stats.get('max_drawdown',0)*100:.1f}% | "
                          f"n={stats.get('count', 0)}")
                except Exception:
                    print(f"       HMM state {hmm_state} -> {label} | "
                          f"z_ret={mean_zscored:.3f} | "
                          f"share={stats.get('time_share',0)*100:.1f}% | "
                          f"maxDD={stats.get('max_drawdown',0)*100:.1f}% | "
                          f"n={stats.get('count', 0)}")
            else:
                print(f"       HMM state {hmm_state} -> {label} | "
                      f"μ_ret={mean_zscored*100:.3f}% | "
                      f"share={stats.get('time_share',0)*100:.1f}% | "
                      f"maxDD={stats.get('max_drawdown',0)*100:.1f}% | "
                      f"n={stats.get('count', 0)}")

    @staticmethod
    def _state_signature(model) -> np.ndarray:
        """Emission fingerprint of each hidden state: [means | std-devs]."""
        covars = np.asarray(model.covars_)
        stds = np.sqrt(np.einsum('kii->ki', covars)) if covars.ndim == 3 else np.sqrt(covars)
        return np.hstack([np.asarray(model.means_), stds])

    def fit(self, benchmark_returns: pd.Series,
            auto_select_regimes: bool = False,
            n_ensemble: int = 10) -> 'MarketRegimeHMM':
        obs = self._prepare_observations(benchmark_returns)

        if auto_select_regimes:
            self.n_regimes = self._select_n_regimes(obs)

        print(f"\n[INFO] Training HMM ensemble ({n_ensemble} models, {self.n_regimes} regimes)...")

        cov_configs = [
            {"covariance_type": "full", "min_covar": 1e-4},
            {"covariance_type": "full", "min_covar": 1e-3},
            {"covariance_type": "full", "min_covar": 1e-2},
            {"covariance_type": "diag", "min_covar": 1e-4},
            {"covariance_type": "diag", "min_covar": 1e-2},
        ]
        ensemble_models = []
        ensemble_lls    = []
        ensemble_probs  = []
        last_error = None

        for seed in range(n_ensemble):
            for cfg in cov_configs:
                try:
                    m = GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=cfg["covariance_type"],
                        n_iter=self.n_iter,
                        min_covar=cfg["min_covar"],
                        random_state=self.random_state + seed,
                        tol=1e-4
                    )
                    m.fit(obs)
                    ll = m.score(obs)
                    probs = m.predict_proba(obs)
                    ensemble_models.append(m)
                    ensemble_lls.append(ll)
                    ensemble_probs.append(probs)
                    break
                except Exception as e:
                    last_error = e
                    continue

        if not ensemble_models:
            raise RuntimeError(
                "[ERROR] HMM did not converge with any configuration.\n"
                f"Last error: {last_error}\n"
                "Try: reducing n_regimes to 2, or increasing period_years to 7."
            )

        print(f"  [OK] {len(ensemble_models)}/{n_ensemble} models converged")
        print(f"  Log-likelihood range: [{min(ensemble_lls):.2f}, {max(ensemble_lls):.2f}]")

        best_idx = int(np.argmax(ensemble_lls))
        self.model = ensemble_models[best_idx]

        ll_arr     = np.array(ensemble_lls)
        ll_shifted = ll_arr - ll_arr.max()   # numerical stability
        weights_ll = np.exp(ll_shifted)
        weights_ll /= weights_ll.sum()

        # Align every ensemble member to the BEST model's state labelling
        # (Hungarian matching on emission parameters). Column k of regime_probs
        # therefore always refers to the same raw state k used by transmat_,
        # regime_sequence and regime_mapping. (Previously columns were sorted by
        # mean return, which does not coincide with the raw state ids.)
        ref_sig = self._state_signature(self.model)
        aligned_probs = []
        for m, probs in zip(ensemble_models, ensemble_probs):
            if m is self.model:
                aligned_probs.append(probs)
                continue
            cost = np.linalg.norm(
                ref_sig[:, None, :] - self._state_signature(m)[None, :, :], axis=2)
            _, col_ind = linear_sum_assignment(cost)
            aligned_probs.append(probs[:, col_ind])

        stacked = np.stack(aligned_probs, axis=0)          # (n_models, T, n_regimes)
        ensemble_avg = np.einsum("i,ijk->jk", weights_ll, stacked)

        floor_pct = self.regime_prob_floor
        n_reg     = ensemble_avg.shape[1]
        # Affine floor p' = floor + (1 - K*floor) * p: rows still sum to 1 AND p' >= floor exactly
        # (clip + renormalise could push entries below the floor)
        self.regime_probs = floor_pct + (1.0 - n_reg * floor_pct) * ensemble_avg

        self.regime_sequence = self.model.predict(obs)

        agreement = float(np.mean(self.regime_sequence == np.argmax(self.regime_probs, axis=1)))
        print(f"  [CHECK] Viterbi path vs ensemble posterior agreement: {agreement*100:.1f}%")
        if agreement < 0.80:
            print("  [WARN] Low agreement: regime labelling / probabilities may be unreliable.")

        self.is_fitted = True
        self.benchmark_index = benchmark_returns.index[-len(obs):]
        self.benchmark_returns_aligned = benchmark_returns.values[-len(obs):]

        self._map_regimes_economically()

        ll_best = self.model.score(obs)
        print(f"[OK] HMM ensemble trained — Best LL: {ll_best:.4f} | "
              f"Prob floor: {floor_pct*100:.0f}% per regime")

        return self

    def _run_validation_after_fit(self, benchmark_returns: pd.Series) -> None:
        """Called automatically by the pipeline after fit — do not call directly."""
        try:
            self.validate_hmm_quality(benchmark_returns)
        except Exception as e:
            print(f"  [WARN] HMM validation not completed: {e}")

    def get_current_regime_probs(self) -> Dict[str, float]:
        """
        Posterior regime probabilities at the last observation, keyed by regime label.

        At the final time step the smoothed posterior coincides with the filtered
        (Hamilton) probability, so no look-ahead is involved for "today".
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Run .fit() first.")

        last_probs = self.regime_probs[-1]
        result = {}
        for hmm_state, prob in enumerate(last_probs):
            economic_state = self.regime_mapping.get(hmm_state, hmm_state)
            label = self.REGIME_LABELS.get(economic_state, f"Regime {economic_state}")
            result[label] = float(prob)

        return result

    def get_regime_statistics(self,
                              etf_returns: pd.DataFrame) -> Dict[int, Dict]:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")

        # Align by DATE: ETFs trade on different exchange calendars, so taking the
        # last N regime labels would shift labels against returns by several days.
        regime_series = pd.Series(self.regime_sequence, index=self.benchmark_index)
        common_idx = regime_series.index.intersection(etf_returns.index)
        etf_data = etf_returns.loc[common_idx].dropna()
        reg_seq = regime_series.loc[etf_data.index].values

        # Annualized priors by sector — use CONFIG if available
        default_priors = {
            "MSCI World":         0.08,   # 8% p.a.
            "Fixed Income":       0.035,  # 3.5% p.a.
            "Commodities":        0.05,   # 5% p.a.
            "Military & Defence": 0.09,   # 9% p.a.
            "Tech & AI":          0.10,   # 10% p.a.
            "Monetary":           0.03,   # 3% p.a. — v5 FIX: renamed from "Monetary ETF"
            "Financial Sector":   0.08,   # 8% p.a.
            "Emerging Markets":   0.09,   # 9% p.a.
            "Asia Pacific":       0.07,   # 7% p.a.
            "Utility and index":  0.085,  # 8.5% p.a.
            "Materials":          0.08,   # 8% p.a.
            "DEFAULT":            0.07,   # 7% fallback
        }
        end_date = etf_data.index[-1] if len(etf_data) else None
        prior_schedule = getattr(self, "_prior_schedule", None)
        strict_prior_schedule = bool(getattr(self, "_strict_prior_schedule", False))
        selected_priors = None
        if prior_schedule and end_date is not None:
            try:
                candidates = []
                for item in prior_schedule:
                    d = pd.Timestamp(item.get("effective_date"))
                    if d <= pd.Timestamp(end_date):
                        candidates.append((d, item.get("priors", {})))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    selected_priors = dict(candidates[-1][1])
                    print(f"  [INFO] As-of prior schedule active ({candidates[-1][0].date()})")
                elif strict_prior_schedule:
                    raise ValueError("No prior schedule available at the window date.")
            except Exception as e:
                if strict_prior_schedule:
                    raise
                print(f"  [WARN] prior_schedule not applied ({e}) — using static priors")
        SECTOR_PRIORS_ANNUAL = selected_priors or getattr(self, '_sector_priors', None) or default_priors

        ticker_list = list(etf_data.columns)

        use_arith = getattr(self, '_mu_convention', 'arithmetic') == 'arithmetic'

        def annual_mean(frame):
            """Annualised mean return: arithmetic E[R] (average of simple returns) or log E[ln(1+R)]."""
            vals = np.expm1(frame.values) if use_arith else frame.values
            return vals.mean(axis=0) * 252

        mu_prior_annual = np.array([
            SECTOR_PRIORS_ANNUAL.get(
                getattr(self, '_sector_map', {}).get(t, ''),
                SECTOR_PRIORS_ANNUAL.get("DEFAULT", 0.07)
            )
            for t in ticker_list
        ])

        if not hasattr(self, '_sector_map') or not self._sector_map:
            mu_prior_annual  = annual_mean(etf_data)
            print("  [INFO] Sector map not available — using historical prior")
        else:
            print(f"  [INFO] Forward-looking sector priors active "
                  f"(range: {mu_prior_annual.min()*100:.1f}%-{mu_prior_annual.max()*100:.1f}%)")

        n_total_etf_obs = len(etf_data)
        n_assets_js = len(ticker_list)
        JS_C, js_tier, js_floored = _js_base_intensity(n_total_etf_obs, n_assets_js)
        print(f"  [INFO] {js_tier} ({n_total_etf_obs} obs) -- JS_C={JS_C:.0f}")
        if js_floored:
            print(f"  [INFO] JS_C set by the N-aware floor (N={n_assets_js}, (N-2)×2={JS_C:.0f})")

        MIN_OBS_REGIME = 15

        regime_stats = {}
        print("\n[INFO] Estimating (μ, Σ) per regime with Ledoit-Wolf + James-Stein:")

        for hmm_state in range(self.n_regimes):
            economic_state = self.regime_mapping.get(hmm_state, hmm_state)
            label = self.REGIME_LABELS.get(economic_state, f"Regime {economic_state}")

            mask = reg_seq == hmm_state
            regime_data = etf_data[mask]
            n_obs = len(regime_data)

            if n_obs < MIN_OBS_REGIME:
                print(f"  [{label}] WARN: only {n_obs} obs (<{MIN_OBS_REGIME}) — "
                      f"using the global series + strong shrinkage")
                regime_data_fit = etf_data
            else:
                regime_data_fit = regime_data

            N_REF = 60
            js_c_adaptive = JS_C * max(1.0, np.sqrt(N_REF / max(n_obs, 1)))
            gamma = js_c_adaptive / (js_c_adaptive + n_obs)
            # Annualized sample mean of the regime (or of the full series if the regime has too few obs)
            mu_sample_annual = annual_mean(regime_data_fit)
            mu_shrunk_annual = (1 - gamma) * mu_sample_annual + gamma * mu_prior_annual

            sigma_daily_lw, lw_shrinkage = _ledoit_wolf_cov(
                regime_data_fit.values, getattr(self, '_lw_preserve_var', True))
            sigma_annual_lw = sigma_daily_lw * 252

            vol_avg_annual = float(np.sqrt(np.diag(sigma_annual_lw)).mean())
            RF_ANNUAL = getattr(self, '_risk_free', 0.03)
            MAX_SHARPE_CAP = 1.8
            MU_CLIP_LOW  = -0.50
            MU_CLIP_HIGH = min(RF_ANNUAL + MAX_SHARPE_CAP * vol_avg_annual, 0.20)
            mu_clipped = np.clip(mu_shrunk_annual, MU_CLIP_LOW, MU_CLIP_HIGH)

            n_clipped = np.sum(
                (mu_shrunk_annual < MU_CLIP_LOW) | (mu_shrunk_annual > MU_CLIP_HIGH)
            )
            if n_clipped > 0:
                print(f"  [{label}] INFO: {n_clipped} assets clipped "
                      f"(mu_clip_high={MU_CLIP_HIGH*100:.1f}%, vol_avg={vol_avg_annual*100:.1f}%)")

            regime_mu_floors = getattr(self, '_regime_mu_floors', {})
            mu_floor_val = regime_mu_floors.get(label, None)
            mu_floor_applied = None
            if mu_floor_val is not None:
                n_floored = int(np.sum(mu_clipped < mu_floor_val))
                if n_floored > 0:
                    print(f"  [{label}] INFO: {n_floored} assets floored at "
                          f"{mu_floor_val*100:.1f}% p.a. (regime_mu_floor)")
                mu_clipped = np.maximum(mu_clipped, mu_floor_val)
            mu_floor_applied = mu_floor_val

            use_mp = getattr(self, '_use_mp_denoising', True)
            if use_mp:
                sigma_annual, mp_info = _mp_denoise_covariance(
                    sigma_annual_lw, regime_data_fit.values
                )
                mp_tag = (f"MP[sig={mp_info['n_signal']},α_lw={mp_info['alpha_lw']:.2f}]"
                          if not mp_info.get('skipped')
                          else "MP[skip]")
            else:
                sigma_annual = sigma_annual_lw
                mp_info = {'skipped': True}
                mp_tag  = "MP[off]"

            sigma_daily = sigma_annual / 252

            regime_stats[economic_state] = {
                'label':             label,
                'mu':                mu_clipped,
                'mu_raw':            mu_sample_annual,
                'mu_bounds':         (MU_CLIP_LOW, MU_CLIP_HIGH, mu_floor_applied),
                'sigma':             sigma_annual,      # LW + MP blended
                'sigma_lw':          sigma_annual_lw,   # LW ONLY
                'n_obs':             n_obs,
                'shrinkage_sigma':   lw_shrinkage,
                'shrinkage_mu':      float(gamma),
                'mp_info':           mp_info,
                'mean_return_daily': float(mu_clipped.mean() / 252),
                'mean_vol_daily':    float(np.sqrt(np.diag(sigma_daily)).mean())
            }

            print(f"  [{label}] n={n_obs} obs | "
                  f"μ_raw={mu_sample_annual.mean()*100:.1f}% → "
                  f"μ_JS={mu_clipped.mean()*100:.1f}% p.a. "
                  f"(γ={gamma:.2f}) | "
                  f"σ={np.sqrt(np.diag(sigma_annual)).mean()*100:.1f}% p.a. | "
                  f"LW={lw_shrinkage:.3f} | {mp_tag}")

        return regime_stats

    def apply_momentum_tilt(self,
                             etf_returns: pd.DataFrame,
                             mu_by_regime: dict,
                             momentum_window_long: int = 252,
                             momentum_window_short: int = 21,
                             tilt_strength: float = 0.15) -> dict:

        if len(etf_returns) < 60:
            print("  [WARN] Insufficient data for momentum (< 60 days) — tilt skipped")
            return mu_by_regime
        momentum_window_long  = min(momentum_window_long,  len(etf_returns) - momentum_window_short - 1)
        momentum_window_short = min(momentum_window_short, max(5, len(etf_returns) // 5))
        print(f"  Adapted windows: long={momentum_window_long}d, short={momentum_window_short}d")

        recent_prices = np.exp(etf_returns.cumsum())   # log returns -> price index
        if len(recent_prices) > momentum_window_long + momentum_window_short:
            ret_long  = recent_prices.iloc[-momentum_window_short] / recent_prices.iloc[-(momentum_window_long + momentum_window_short)] - 1
        else:
            ret_long  = recent_prices.iloc[-1] / recent_prices.iloc[0] - 1

        momentum_score = ret_long.values

        mom_mean = momentum_score.mean()
        mom_std  = momentum_score.std()
        if mom_std < 1e-8:
            return mu_by_regime
        momentum_z = (momentum_score - mom_mean) / mom_std
        MOMENTUM_Z_CAP = 2.0
        n_clipped_mom = int(np.sum(np.abs(momentum_z) > MOMENTUM_Z_CAP))
        if n_clipped_mom > 0:
            print(f"  [MOMENTUM] Winsorizing {n_clipped_mom} outlier z-scores (|z|>{MOMENTUM_Z_CAP})")
        momentum_z = np.clip(momentum_z, -MOMENTUM_Z_CAP, MOMENTUM_Z_CAP)

        avg_mu_dispersion = float(np.mean([np.std(s['mu']) for s in mu_by_regime.values()]))
        print(f"\n[INFO] Momentum Tilt (strength={tilt_strength}, "
              f"avg cross-sectional μ dispersion={avg_mu_dispersion*100:.2f}%):")
        for i, ticker in enumerate(etf_returns.columns):
            applied = momentum_z[i] * tilt_strength * avg_mu_dispersion
            print(f"  {ticker:12s} z={momentum_z[i]:+.2f} | applied μ tilt: {applied*100:+.2f}% p.a.")

        tilted_mu = {}
        for regime_key, stats in mu_by_regime.items():
            mu_original = stats['mu'].copy()
            tilt_vector = momentum_z * tilt_strength * np.std(mu_original)
            mu_tilted = mu_original + tilt_vector
            # the tilt must not break the safety bounds applied during estimation
            lo, hi, floor = stats.get('mu_bounds', (-np.inf, np.inf, None))
            mu_tilted = np.clip(mu_tilted, lo, hi)
            if floor is not None:
                mu_tilted = np.maximum(mu_tilted, floor)
            tilted_stats = dict(stats)
            tilted_stats['mu'] = mu_tilted
            tilted_mu[regime_key] = tilted_stats

        return tilted_mu

    def validate_hmm_quality(self, benchmark_returns: pd.Series,
                              train_ratio: float = 0.70) -> Dict:
        if not self.is_fitted:
            print("  [WARN] HMM not fitted yet — validation skipped")
            return {}

        n = len(benchmark_returns)
        n_train = int(n * train_ratio)
        train_ret = benchmark_returns.iloc[:n_train]
        test_ret  = benchmark_returns.iloc[n_train:]

        print(f"\n[INFO] HMM validation (Train/Test split {int(train_ratio*100)}/{int((1-train_ratio)*100)}):")
        print(f"       Train: {train_ret.index[0].strftime('%d/%m/%Y')} -> "
              f"{train_ret.index[-1].strftime('%d/%m/%Y')} ({n_train} days)")
        print(f"       Test:  {test_ret.index[0].strftime('%d/%m/%Y')} -> "
              f"{test_ret.index[-1].strftime('%d/%m/%Y')} ({n - n_train} days)")

        report = {}
        orig_scaler = self.scaler

        try:
            obs_train = self._prepare_observations(train_ret, fit_scaler=True)
            obs_test  = self._prepare_observations(test_ret,  fit_scaler=False)
            val_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.model.covariance_type,
                n_iter=self.n_iter, min_covar=1e-3,
                random_state=self.random_state, tol=1e-4
            )
            val_model.fit(obs_train)
            ll_train = val_model.score(obs_train) / len(obs_train)
            ll_test  = val_model.score(obs_test)  / len(obs_test)
            ll_gap   = abs(ll_train - ll_test) / abs(ll_train) * 100
            report['ll_train']   = ll_train
            report['ll_test']    = ll_test
            report['ll_gap_pct'] = ll_gap
            quality = "OK" if ll_gap < 10 else ("WARN" if ll_gap < 20 else "HIGH OVERFITTING")
            print(f"       LL/obs Train: {ll_train:.4f} | Test: {ll_test:.4f} | "
                  f"Gap: {ll_gap:.1f}% [{quality}]")
        except Exception as e:
            print(f"       [WARN] Train/Test LL could not be computed: {e}")
            report['ll_gap_pct'] = None

        # Silhouette Score
        try:
            obs_full = self._prepare_observations(benchmark_returns, fit_scaler=True)
            regime_seq = self.model.predict(obs_full)
            unique_labels = np.unique(regime_seq)
            if len(unique_labels) >= 2:
                sil = sk_silhouette_score(obs_full, regime_seq,
                                          sample_size=min(1000, len(obs_full)),
                                          random_state=self.random_state)
                report['silhouette'] = sil
                if sil > 0.30:
                    sil_label = "EXCELLENT — well-separated regimes"
                elif sil > 0.10:
                    sil_label = "ACCEPTABLE — normal for financial data"
                else:
                    sil_label = "LOW — overlapping regimes, μ_k less reliable"
                print(f"       Silhouette Score: {sil:.4f} [{sil_label}]")
            else:
                print("       [WARN] Silhouette: fewer than 2 distinct regimes")
                report['silhouette'] = None
        except Exception as e:
            print(f"       [WARN] Silhouette: {e}")
            report['silhouette'] = None

        print("       Observations per regime:")
        obs_counts = {}
        for s in range(self.n_regimes):
            eco   = self.regime_mapping.get(s, s)
            label = self.REGIME_LABELS.get(eco, f"R{s}")
            count = int((self.regime_sequence == s).sum())
            pct   = count / len(self.regime_sequence) * 100
            obs_counts[label] = count
            note  = "OK" if count >= 30 else "WARN: <30 obs, unstable mu"
            print(f"         {label:20s}: {count:4d} obs ({pct:.1f}%) [{note}]")
        report['obs_per_regime'] = obs_counts

        self.scaler = orig_scaler
        return report

    def print_transition_matrix(self) -> None:
        if not self.is_fitted:
            return
        print("\n[INFO] Regime transition matrix:")
        print("       (row = current state, column = next state)")
        header = "       " + "  ".join(
            [f"{self.REGIME_LABELS.get(self.regime_mapping.get(s,s), s):>15}" for s in range(self.n_regimes)]
        )
        print(header)
        for i in range(self.n_regimes):
            row_label = self.REGIME_LABELS.get(self.regime_mapping.get(i,i), str(i))
            row = "  ".join([f"{self.model.transmat_[i][j]:>15.4f}" for j in range(self.n_regimes)])
            print(f"  {row_label:>15}: {row}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3:  MVO-REGIME OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

class RegimeMVOptimizer:

    def __init__(self,
                 risk_free_rate: float = 0.03,
                 min_weight: float = 0.0,
                 max_weight: float = 0.40,
                 sector_constraints: Optional[Dict] = None,
                 erc_blend: float = 0.0):
        self.rf = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sector_constraints = sector_constraints or {}
        self.erc_blend = erc_blend

    def _compute_erc_weights(self, sigma: np.ndarray, n_assets: int) -> np.ndarray:
        def erc_objective(w):
            w = np.maximum(w, 1e-8)
            port_var = w @ sigma @ w
            if port_var < 1e-12:
                return 1e10
            mctr = sigma @ w  # vector (n_assets,)
            # Risk contribution: w_i * MCTR_i / port_var
            rc = w * mctr / port_var
            return float(np.sum((rc - rc.mean())**2))

        w0 = np.ones(n_assets) / n_assets
        bounds = [(self.min_weight, self.max_weight)] * n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        best_w = w0.copy()
        best_val = np.inf
        for seed in range(5):
            w_init = np.random.default_rng(seed * 13).dirichlet(np.ones(n_assets))
            w_init = np.clip(w_init, self.min_weight, self.max_weight)
            w_init /= w_init.sum()
            try:
                res = minimize(erc_objective, w_init, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-10})
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_w = res.x
            except Exception:
                continue

        if not np.isfinite(best_val):
            print("         [WARN] ERC solver did not converge — using equal weights")
        best_w = np.clip(best_w, 0, None)
        best_w /= best_w.sum()
        return best_w

    def _sharpe_neg(self, weights: np.ndarray,
                    mu: np.ndarray, sigma: np.ndarray) -> float:
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        if port_vol < 1e-10:
            return 1e10
        return -(port_return - self.rf) / port_vol

    def optimize_scipy(self, mu: np.ndarray, sigma: np.ndarray,
                       n_assets: int, ticker_names: List[str],
                       sector_map: Optional[Dict] = None) -> np.ndarray:

        constraints = _slsqp_constraints(ticker_names, sector_map, self.sector_constraints)

        bounds = [(self.min_weight, self.max_weight)] * n_assets
        best_result = None
        best_sharpe = -np.inf

        for seed in range(10):
            w_init = np.random.default_rng(seed * 7).dirichlet(np.ones(n_assets))
            w_init = np.clip(w_init, self.min_weight, self.max_weight)
            w_init /= w_init.sum()

            try:
                result = minimize(
                    self._sharpe_neg, w_init,
                    args=(mu, sigma),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
                if result.success and -result.fun > best_sharpe:
                    best_sharpe = -result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None or not best_result.success:
            print("  [WARN] Optimization did not converge — using equal weight")
            return np.ones(n_assets) / n_assets

        return best_result.x

    def _cvx_parts(self, mu, sigma, n_assets, ticker_names, sector_map, max_variance=None):
        """CVXPY building blocks shared by every cvxpy-based mode."""
        w = cp.Variable(n_assets)
        port_return = mu @ w
        port_risk = cp.quad_form(w, sigma)
        constraints_list = [cp.sum(w) == 1, w >= self.min_weight, w <= self.max_weight]
        if max_variance is not None:
            constraints_list.append(port_risk <= max_variance)
        for idx, s_min, s_max in _sector_groups(ticker_names, sector_map, self.sector_constraints):
            constraints_list.append(cp.sum(w[idx]) >= s_min)
            constraints_list.append(cp.sum(w[idx]) <= s_max)
        return w, port_return, port_risk, constraints_list

    def optimize_cvxpy(self, mu: np.ndarray, sigma: np.ndarray,
                       n_assets: int, ticker_names: List[str],
                       sector_map: Optional[Dict] = None) -> np.ndarray:

        w, port_return, port_risk, constraints_list = self._cvx_parts(
            mu, sigma, n_assets, ticker_names, sector_map)

        best_weights = None
        best_sharpe = -np.inf

        lam = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Maximize(port_return - lam * port_risk), constraints_list)   # compiled once (DPP)
        for lambda_risk in np.logspace(-4, 4, 120):
            lam.value = lambda_risk
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
                if w.value is not None:
                    weights = np.array(w.value).flatten()
                    weights = np.clip(weights, 0, None)
                    weights /= weights.sum()
                    p_ret = np.dot(weights, mu)
                    p_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                    sharpe = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else -1e10
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = weights
            except Exception:
                continue

        if best_weights is None:
            print("  [WARN] CVXPY found no solution — using SciPy")
            return self.optimize_scipy(mu, sigma, n_assets, ticker_names, sector_map)

        scipy_weights = self.optimize_scipy(mu, sigma, n_assets, ticker_names, sector_map)
        p_ret_cvx = np.dot(best_weights, mu)
        p_vol_cvx = np.sqrt(np.dot(best_weights, np.dot(sigma, best_weights)))
        sharpe_cvx = (p_ret_cvx - self.rf) / p_vol_cvx if p_vol_cvx > 1e-10 else -1e10

        p_ret_sp = np.dot(scipy_weights, mu)
        p_vol_sp = np.sqrt(np.dot(scipy_weights, np.dot(sigma, scipy_weights)))
        sharpe_sp = (p_ret_sp - self.rf) / p_vol_sp if p_vol_sp > 1e-10 else -1e10
        if sharpe_sp > sharpe_cvx:
            return scipy_weights

        return best_weights

    def _apply_erc_blend_if_enabled(self,
                                    weights: np.ndarray,
                                    sigma: np.ndarray,
                                    n_assets: int,
                                    mu: np.ndarray) -> np.ndarray:
        if self.erc_blend <= 0.0:
            return weights
        try:
            erc_weights = self._compute_erc_weights(sigma, n_assets)
            blended = (1 - self.erc_blend) * weights + self.erc_blend * erc_weights
            blended = np.clip(blended, 0, None)
            blended /= blended.sum()
            p_ret_blend = np.dot(blended, mu)
            p_vol_blend = np.sqrt(np.dot(blended, np.dot(sigma, blended)))
            sharpe_blend = (p_ret_blend - self.rf) / p_vol_blend if p_vol_blend > 1e-10 else 0.0
            print(f"         [ERC blend={self.erc_blend}] "
                  f"Return={p_ret_blend*100:.2f}% | "
                  f"Vol={p_vol_blend*100:.2f}% | Sharpe={sharpe_blend:.3f}")
            return blended
        except Exception as e:
            print(f"         [WARN] ERC blend failed ({e}) — using pure MVO weights")
            return weights

    def optimize_for_regime(self, mu: np.ndarray, sigma: np.ndarray,
                            n_assets: int, ticker_names: List[str],
                            regime_label: str,
                            sector_map: Optional[Dict] = None,
                            return_tilt: float = 0.0,
                            vol_target: Optional[float] = None) -> np.ndarray:

        print(f"  Optimizing regime [{regime_label}]...", end=" ")

        # --- VOL TARGET mode: max return s.t. vol <= target ---
        if vol_target is not None and CVXPY_AVAILABLE:
            w, port_return, port_risk, constraints_list = self._cvx_parts(
                mu, sigma, n_assets, ticker_names, sector_map, max_variance=vol_target**2)

            prob = cp.Problem(cp.Maximize(port_return), constraints_list)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
                if w.value is not None:
                    weights = np.clip(np.array(w.value).flatten(), 0, None)
                    weights /= weights.sum()
                    weights = self._apply_erc_blend_if_enabled(weights, sigma, n_assets, mu)
                    p_ret = np.dot(weights, mu)
                    p_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                    sharpe = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else 0.0
                    print(f"Return={p_ret*100:.2f}% | Vol={p_vol*100:.2f}% | Sharpe={sharpe:.3f} [vol-target]")
                    return weights
                print(f"\n         [WARN] vol_target={vol_target} is infeasible in this regime "
                      f"(solver status: {prob.status}) — using the minimum-variance portfolio instead")
                w_mv, _, risk_mv, cons_mv = self._cvx_parts(mu, sigma, n_assets, ticker_names, sector_map)
                cp.Problem(cp.Minimize(risk_mv), cons_mv).solve(solver=cp.CLARABEL, verbose=False)
                if w_mv.value is not None:
                    weights = np.clip(np.array(w_mv.value).flatten(), 0, None)
                    weights /= weights.sum()
                    p_ret = np.dot(weights, mu)
                    p_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                    print(f"         Return={p_ret*100:.2f}% | Vol={p_vol*100:.2f}% [min-variance fallback]")
                    return weights
            except Exception as e:
                print(f"\n         [WARN] vol_target solve failed ({e}) — falling back to the next mode")

        if return_tilt > 0 and CVXPY_AVAILABLE:
            w, port_return, port_risk, constraints_list = self._cvx_parts(
                mu, sigma, n_assets, ticker_names, sector_map)

            best_weights = None
            best_score   = -np.inf
            for lambda_risk in np.logspace(-2, 2, 40):
                sharpe_obj = port_return - lambda_risk * port_risk
                return_obj = port_return
                objective  = cp.Maximize((1 - return_tilt) * sharpe_obj
                                         + return_tilt * return_obj)
                prob = cp.Problem(objective, constraints_list)
                try:
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                    if w.value is not None:
                        ww = np.clip(np.array(w.value).flatten(), 0, None)
                        ww /= ww.sum()
                        p_ret = np.dot(ww, mu)
                        p_vol = np.sqrt(np.dot(ww, np.dot(sigma, ww)))
                        sharpe = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else -1e10
                        score = (1 - return_tilt) * sharpe + return_tilt * p_ret * 10
                        if score > best_score:
                            best_score   = score
                            best_weights = ww
                except Exception:
                    continue

            if best_weights is not None:
                best_weights = self._apply_erc_blend_if_enabled(best_weights, sigma, n_assets, mu)
                p_ret = np.dot(best_weights, mu)
                p_vol = np.sqrt(np.dot(best_weights, np.dot(sigma, best_weights)))
                sharpe = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else 0.0
                print(f"Return={p_ret*100:.2f}% | Vol={p_vol*100:.2f}% | Sharpe={sharpe:.3f} [tilt={return_tilt}]")
                return best_weights

        if return_tilt > 0:
            print(f"\n         [WARN] return_tilt={return_tilt} produced no solution — using plain max Sharpe")

        # ---DEFAULT MODE: ONLY max Sharpe ---
        if CVXPY_AVAILABLE:
            weights = self.optimize_cvxpy(mu, sigma, n_assets, ticker_names, sector_map)
        else:
            weights = self.optimize_scipy(mu, sigma, n_assets, ticker_names, sector_map)

        p_ret = np.dot(weights, mu)
        p_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
        sharpe = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else 0.0
        print(f"Return={p_ret*100:.2f}% | Vol={p_vol*100:.2f}% | Sharpe={sharpe:.3f}")

        weights = self._apply_erc_blend_if_enabled(weights, sigma, n_assets, mu)

        return weights


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: REGIME PROBABILITY WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════

class RegimeProbabilityWeighter:

    MARKET_ANALYSIS_WEIGHTS = {
        "Bull": 0.58,
        "Sideways/Kangaroo": 0.27,
        "Bear": 0.15
    }

    def __init__(self, weighting_mode: str = "blend",
                 blend_alpha: float = 0.6,
                 custom_weights: Optional[Dict[str, float]] = None):
        self.weighting_mode = weighting_mode
        self.blend_alpha = blend_alpha
        self.custom_weights = custom_weights

    def _get_stationary_distribution(self, hmm_model) -> Dict[str, float]:
        A = hmm_model.model.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        result = {}
        for hmm_state, prob in enumerate(stationary):
            economic_state = hmm_model.regime_mapping.get(hmm_state, hmm_state)
            label = MarketRegimeHMM.REGIME_LABELS.get(economic_state, f"Regime {economic_state}")
            result[label] = max(0.0, float(prob))

        total = sum(result.values())
        return {k: v / total for k, v in result.items()}

    def compute_final_weights(self,
                              regime_portfolios: Dict[str, np.ndarray],
                              hmm_model: MarketRegimeHMM) -> Tuple[np.ndarray, Dict]:
        current_probs = hmm_model.get_current_regime_probs()
        stationary_probs = self._get_stationary_distribution(hmm_model)

        if self.weighting_mode == "current":
            regime_weights = current_probs
            mode_desc = "Instantaneous probabilities (Hamilton filter)"

        elif self.weighting_mode == "stationary":
            regime_weights = stationary_probs
            mode_desc = "Stationary distribution (long run)"

        elif self.weighting_mode == "custom" and self.custom_weights:
            total = sum(self.custom_weights.values())
            regime_weights = {k: v / total for k, v in self.custom_weights.items()}
            mode_desc = "User-defined weights"

        elif self.weighting_mode == "market_analysis":
            regime_weights = self.MARKET_ANALYSIS_WEIGHTS
            mode_desc = "Static macro view (set in Mar 2025 — review before use)"

        else:
            regime_weights = {}
            all_labels = set(list(current_probs.keys()) + list(stationary_probs.keys()))
            for label in all_labels:
                c = current_probs.get(label, 0.0)
                s = stationary_probs.get(label, 0.0)
                regime_weights[label] = self.blend_alpha * c + (1 - self.blend_alpha) * s
            total = sum(regime_weights.values())
            regime_weights = {k: v / total for k, v in regime_weights.items()}
            mode_desc = f"Blend (α={self.blend_alpha}: {self.blend_alpha*100:.0f}% current + {(1-self.blend_alpha)*100:.0f}% stationary)"

        print(f"\n[INFO] Regime weighting mode: {mode_desc}")
        print("       Probabilities assigned to regimes:")
        for label, prob in regime_weights.items():
            print(f"         {label}: {prob*100:.1f}%")

        n_assets = None
        for p in regime_portfolios.values():
            if p is not None:
                n_assets = len(p)
                break

        if n_assets is None:
            raise ValueError("No regime portfolio was computed.")

        final_weights = np.zeros(n_assets)
        total_prob_used = 0.0

        for regime_label, portfolio_weights in regime_portfolios.items():
            prob = regime_weights.get(regime_label, 0.0)
            if portfolio_weights is not None and prob > 0:
                final_weights += prob * portfolio_weights
                total_prob_used += prob

        if total_prob_used > 0:
            final_weights /= total_prob_used

        probability_report = {
            'mode': mode_desc,
            'current_probs': current_probs,
            'stationary_probs': stationary_probs,
            'applied_probs': regime_weights,
            'regime_portfolios': regime_portfolios
        }

        return final_weights, probability_report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4B: JOINT STOCHASTIC PROGRAMMING OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class JointStochasticOptimizer:
    def __init__(self,
                 n_sequences: int = 1000,
                 seq_length: int = 22,
                 objective: str = "max_avg_sharpe",
                 risk_free_rate: float = 0.03,
                 min_weight: float = 0.02,
                 max_weight: float = 0.40,
                 sector_constraints: Optional[Dict] = None,
                 erc_blend: float = 0.0,
                 var_confidence: float = 0.05,
                 use_nested_cv: bool = True,
                 nested_cv_ratio: float = 0.70,
                 l2_turnover_penalty: float = 0.10,
                 prev_weights: Optional[np.ndarray] = None,
                 return_tilt: float = 0.0,
                 transaction_cost_bps: float = 5.0,
                 cvar_alpha: float = 0.05):
        if objective not in ("max_avg_sharpe", "min_avg_var", "min_avg_cvar"):
            raise ValueError("objective must be 'max_avg_sharpe', 'min_avg_var' or 'min_avg_cvar'")
        self.n_sequences         = n_sequences
        self.seq_length          = seq_length
        self.objective           = objective
        self.rf                  = risk_free_rate
        self.min_weight          = min_weight
        self.max_weight          = max_weight
        self.sector_constraints  = sector_constraints or {}
        self.erc_blend           = erc_blend
        self.var_confidence      = var_confidence
        self.use_nested_cv       = use_nested_cv
        self.nested_cv_ratio     = nested_cv_ratio
        self.l2_turnover_penalty = l2_turnover_penalty
        self.prev_weights        = prev_weights
        if return_tilt < 0.0 or return_tilt > 1.0:
            print(f"  [WARN] return_tilt={return_tilt:.2f} out of range [0,1]. "
                  f"Suggested values: 0.0=pure Sharpe, 0.20=blend, 1.0=pure return. "
                  f"Clamped to {max(0.0, min(1.0, return_tilt)):.2f}.")
        self.return_tilt = max(0.0, min(1.0, return_tilt))  # blend: 0=Sharpe, 1=MaxReturn
        self.transaction_cost_rate = max(0.0, transaction_cost_bps) / 10000.0
        self.cvar_alpha = min(0.20, max(0.001, cvar_alpha))

    def _simulate_sequences(self,
                             trans_mat: np.ndarray,
                             current_state: int,
                             n_states: int,
                             start_probs: Optional[np.ndarray] = None,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Simulate Markov regime paths; returns per-path state occupation counts.

        Vectorised across paths (loop over time only), ~50-100x faster than a
        per-path Python loop for 1000+ paths of 750+ days.
        """
        rng = rng if rng is not None else np.random.default_rng(42)
        n_paths, horizon = self.n_sequences, self.seq_length

        cum_trans = np.cumsum(trans_mat, axis=1)
        cum_trans[:, -1] = 1.0

        if start_probs is not None:
            cum_start = np.cumsum(start_probs)
            cum_start[-1] = 1.0
            state = np.minimum(np.searchsorted(cum_start, rng.random(n_paths)), n_states - 1)
        else:
            state = np.full(n_paths, current_state, dtype=np.int64)

        counts = np.zeros((n_paths, n_states), dtype=np.float64)
        draws = rng.random((horizon, n_paths))
        rows = np.arange(n_paths)
        for t in range(horizon):
            counts[rows, state] += 1.0
            state = np.minimum((draws[t][:, None] > cum_trans[state]).sum(axis=1), n_states - 1)
        return counts

    def _portfolio_stats_from_counts(self,
                                      weights: np.ndarray,
                                      mu_matrix: np.ndarray,
                                      sigma_tensor: np.ndarray,
                                      state_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = weights.flatten()  # (n_assets,) — keep 1-D for np.dot
        L = self.seq_length

        mu_per_state = np.dot(mu_matrix, w)          # (n_states,)
        exp_returns  = np.dot(state_counts, mu_per_state) / L  # (n_sequences,)

        exp_vars = np.zeros(state_counts.shape[0])   # nested CV passes a subset of the paths
        for k in range(sigma_tensor.shape[0]):
            freq_k = state_counts[:, k] / L          # (n_sequences,)
            var_k  = float(np.dot(w, np.dot(sigma_tensor[k], w)))  # scalar
            exp_vars += freq_k * var_k

        exp_vols = np.sqrt(np.maximum(exp_vars, 1e-10))
        return exp_returns, exp_vols

    def _objective_max_avg_sharpe(self,
                                   weights: np.ndarray,
                                   mu_matrix: np.ndarray,
                                   sigma_tensor: np.ndarray,
                                   state_counts: np.ndarray,
                                   w_ref: Optional[np.ndarray] = None) -> float:
        weights = np.clip(weights.flatten(), 0, None)
        total = weights.sum()
        if total < 1e-8:
            return 1e10
        weights = weights / total
        exp_returns, exp_vols = self._portfolio_stats_from_counts(
            weights, mu_matrix, sigma_tensor, state_counts)
        tc = 0.0
        if w_ref is not None and self.transaction_cost_rate > 0:
            tc = self.transaction_cost_rate * float(np.sum(np.abs(weights - w_ref)))
        exp_returns_net = exp_returns - tc
        rf_scaled  = self.rf / 252 * self.seq_length

        mean_ret_agg = float(np.mean(exp_returns_net))
        mean_var_agg = float(np.mean(exp_vols ** 2))          # E[V_k] — aggregated mean variance
        ann          = 252.0 / self.seq_length
        # Annualised Sharpe: keeps the weight of additive terms (return tilt, turnover penalty)
        # independent of the simulation horizon. The rescaling does not change the argmax.
        mean_sharpe  = (mean_ret_agg - rf_scaled) * ann / (float(np.sqrt(max(mean_var_agg * ann, 1e-10))) + 1e-10)
        mean_return  = mean_ret_agg * ann   # annualized

        rt = getattr(self, 'return_tilt', 0.0)
        if rt > 0.0:
            mean_return_ann = mean_return   # already annualized — DO NOT multiply again
            TARGET_VOL_NORM = 0.15  # normalize return to Sharpe scale (15% vol target)
            sharpe_obj = -((1.0 - rt) * mean_sharpe + rt * mean_return_ann / TARGET_VOL_NORM)
        else:
            sharpe_obj = -mean_sharpe

        if self.l2_turnover_penalty > 0.0 and w_ref is not None:
            l2_pen = self.l2_turnover_penalty * float(np.sum((weights - w_ref)**2))
            sharpe_obj += l2_pen
        return sharpe_obj

    def _objective_min_avg_var(self,
                                weights: np.ndarray,
                                mu_matrix: np.ndarray,
                                sigma_tensor: np.ndarray,
                                state_counts: np.ndarray,
                                w_ref: Optional[np.ndarray] = None) -> float:
        weights = np.clip(weights.flatten(), 0, None)
        total = weights.sum()
        if total < 1e-8:
            return 1e10
        weights = weights / total
        exp_returns, exp_vols = self._portfolio_stats_from_counts(
            weights, mu_matrix, sigma_tensor, state_counts)
        tc = 0.0
        if w_ref is not None and self.transaction_cost_rate > 0:
            tc = self.transaction_cost_rate * float(np.sum(np.abs(weights - w_ref)))
        exp_returns_net = exp_returns - tc
        z        = scipy_norm.ppf(self.var_confidence)
        vars_seq = exp_returns_net + z * exp_vols
        var_obj  = -float(np.mean(vars_seq))
        if self.l2_turnover_penalty > 0.0 and w_ref is not None:
            l2_pen  = self.l2_turnover_penalty * float(np.sum((weights - w_ref)**2))
            var_obj += l2_pen
        return var_obj

    def _objective_min_avg_cvar(self,
                                weights: np.ndarray,
                                mu_matrix: np.ndarray,
                                sigma_tensor: np.ndarray,
                                state_counts: np.ndarray,
                                w_ref: Optional[np.ndarray] = None) -> float:
        weights = np.clip(weights.flatten(), 0, None)
        total = weights.sum()
        if total < 1e-8:
            return 1e10
        weights = weights / total
        exp_returns, exp_vols = self._portfolio_stats_from_counts(
            weights, mu_matrix, sigma_tensor, state_counts)

        tc = 0.0
        if w_ref is not None and self.transaction_cost_rate > 0:
            tc = self.transaction_cost_rate * float(np.sum(np.abs(weights - w_ref)))
        exp_returns_net = exp_returns - tc

        alpha = self.cvar_alpha
        z_alpha = scipy_norm.ppf(alpha)
        tail_adj = scipy_norm.pdf(z_alpha) / max(alpha, 1e-8)
        cvar_losses = -exp_returns_net + exp_vols * tail_adj
        cvar_obj = float(np.mean(cvar_losses))
        if self.l2_turnover_penalty > 0.0 and w_ref is not None:
            cvar_obj += self.l2_turnover_penalty * float(np.sum((weights - w_ref)**2))
        return cvar_obj

    def _build_constraints(self, ticker_names: List[str],
                           sector_map: Optional[Dict]) -> list:
        return _slsqp_constraints(ticker_names, sector_map, self.sector_constraints)

    def _lp_feasible_point(self, w_pref: np.ndarray, ticker_names: List[str],
                           sector_map: Optional[Dict]) -> Optional[np.ndarray]:
        """Exact feasible point (asset + sector bounds) via LP, leaning toward w_pref."""
        n = len(w_pref)
        A_ub, b_ub = [], []
        for idx, lo, hi in _sector_groups(ticker_names, sector_map, self.sector_constraints):
            row = np.zeros(n); row[idx] = 1.0
            A_ub += [row, -row]; b_ub += [hi, -lo]
        res = linprog(-w_pref, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=np.ones((1, n)), b_eq=[1.0],
                      bounds=[(self.min_weight, self.max_weight)] * n, method='highs')
        return np.asarray(res.x) if res.success else None

    def _get_feasible_start(self, n_assets: int, ticker_names: List[str],
                             sector_map: Optional[Dict],
                             bounds: list,
                             mu_eff: Optional[np.ndarray] = None) -> np.ndarray:

        if mu_eff is not None and mu_eff.max() > mu_eff.min() + 1e-6:
            mu_pos = np.maximum(mu_eff - mu_eff.min() + 1e-4, 1e-4)
            w = mu_pos / mu_pos.sum()
        else:
            w = np.ones(n_assets) / n_assets
        w = np.clip(w, self.min_weight, self.max_weight)
        w /= w.sum()

        if not sector_map or not self.sector_constraints:
            return w
        try:
            constraints = self._build_constraints(ticker_names, sector_map)
            res = minimize(lambda x: float(np.sum((x - w)**2)), w,
                           bounds=bounds, constraints=constraints,
                           method='SLSQP', options={'maxiter': 2000})
            if res.success:
                ww = np.clip(res.x, 0, None)
                ww /= ww.sum()
                return ww
        except Exception as e:
            print(f"       [WARN] feasible-start projection failed ({e}); trying an exact LP")
        w_lp = self._lp_feasible_point(w, ticker_names, sector_map)
        if w_lp is not None:
            return w_lp
        print("       [WARN] no point satisfies the asset and sector bounds — constraints are inconsistent;"
              " using a best-effort heuristic that may violate them")
        w_fallback = np.full(n_assets, self.min_weight)
        for sector, (s_min, s_max) in self.sector_constraints.items():
            idx = [i for i, t in enumerate(ticker_names)
                   if sector_map.get(t, '') == sector]
            if idx:
                target  = (s_min + s_max) / 2.0
                per_etf = target / len(idx)
                for i in idx:
                    w_fallback[i] = max(self.min_weight, per_etf)
        w_fallback = np.clip(w_fallback, self.min_weight, self.max_weight)
        w_fallback /= w_fallback.sum()
        return w_fallback

    def _enforce_sector_constraints(self, weights: np.ndarray, n_assets: int,
                                     ticker_names: List[str],
                                     sector_map: Optional[Dict],
                                     bounds: list) -> np.ndarray:
        if not sector_map or not self.sector_constraints:
            return weights
        has_violation = False
        for sector, (s_min, s_max) in self.sector_constraints.items():
            idx = [i for i, t in enumerate(ticker_names)
                   if sector_map.get(t, '') == sector]
            if idx:
                sw = float(sum(weights[i] for i in idx))
                if sw < s_min - 1e-4 or sw > s_max + 1e-4:
                    has_violation = True
                    break
        if not has_violation:
            return weights
        print("       [POST] Enforcing sector constraints...", end=" ")
        try:
            constraints = self._build_constraints(ticker_names, sector_map)
            w_target = weights.copy()
            best_res = None
            best_obj = np.inf
            for w_init in [weights.copy(),
                           np.ones(n_assets) / n_assets,
                           self._get_feasible_start(n_assets, ticker_names, sector_map, bounds, mu_eff=w_target)]:
                try:
                    r = minimize(
                        lambda w: float(np.sum((w - w_target) ** 2)),
                        w_init, bounds=bounds, constraints=constraints,
                        method='SLSQP', options={'maxiter': 3000, 'ftol': 1e-12})
                    if r.success and r.fun < best_obj:
                        best_obj = r.fun
                        best_res = r
                except Exception:
                    continue
            if best_res is not None and best_res.success:
                ww = np.clip(best_res.x, 0, None)
                ww /= ww.sum()
                print("OK")
                return ww
            print("WARN — convergence failed, using fallback")
        except Exception as e:
            print(f"WARN ({e})")
        return self._get_feasible_start(n_assets, ticker_names, sector_map, bounds, mu_eff=None)

    def _candidate_starts(self, n_assets: int, mu_matrix: np.ndarray,
                          w_feasible: np.ndarray) -> List[np.ndarray]:
        """Diversified SLSQP starting points: sparse, uniform, mu-proportional, diversified, feasible."""
        mu_eff = np.mean(mu_matrix, axis=0)                       # mean across regimes
        mu_pos = np.maximum(mu_eff - mu_eff.min() + 1e-4, 1e-4)

        def project(w):
            w = np.clip(w, self.min_weight, self.max_weight)
            return w / w.sum()

        starts = []
        for seed in range(4):    # sparse / concentrated starts
            starts.append(project(np.random.default_rng(seed * 17).dirichlet(np.full(n_assets, 0.3))))
        for seed in range(3):    # uniform on the simplex
            starts.append(project(np.random.default_rng(seed * 17 + 100).dirichlet(np.ones(n_assets))))
        for seed in range(4):    # mu-proportional starts with log-normal noise
            rng = np.random.default_rng(seed * 17 + 200)
            starts.append(project(mu_pos * rng.lognormal(0, 0.3, n_assets)))
        for seed in range(4):    # diversified starts
            starts.append(project(np.random.default_rng(seed * 17 + 300).dirichlet(np.full(n_assets, 3.0))))
        starts.append(w_feasible.copy())                          # sector-feasible start
        return starts

    def optimize(self,
                 regime_stats: Dict,
                 hmm_model: 'MarketRegimeHMM',
                 ticker_names: List[str],
                 sector_map: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        n_assets = len(ticker_names)
        n_states = hmm_model.n_regimes

        mu_matrix    = np.zeros((n_states, n_assets))
        sigma_tensor = np.zeros((n_states, n_assets, n_assets))

        hmm_to_eco = hmm_model.regime_mapping

        all_mu_vals    = [s['mu']    for s in regime_stats.values()]
        all_sigma_vals = [s['sigma'] for s in regime_stats.values()]
        mu_global_ann    = np.mean(all_mu_vals,    axis=0) if all_mu_vals    else np.zeros(n_assets)
        sigma_global_ann = np.mean(all_sigma_vals, axis=0) if all_sigma_vals else np.eye(n_assets)*0.04

        for hmm_state in range(n_states):
            eco_state = hmm_to_eco.get(hmm_state, hmm_state)
            if eco_state in regime_stats:
                stats     = regime_stats[eco_state]
                mu_ann    = stats['mu']     # already includes James-Stein + clip
                sigma_ann = stats['sigma']
                mu_matrix[hmm_state]    = mu_ann    / 252 * self.seq_length
                sigma_tensor[hmm_state] = sigma_ann / 252 * self.seq_length
            else:
                mu_matrix[hmm_state]    = mu_global_ann    / 252 * self.seq_length
                sigma_tensor[hmm_state] = sigma_global_ann / 252 * self.seq_length

        mu_clip_lo = -0.80 / 252 * self.seq_length
        mu_clip_hi = +1.50 / 252 * self.seq_length
        mu_matrix = np.clip(mu_matrix, mu_clip_lo, mu_clip_hi)

        trans_mat   = hmm_model.model.transmat_
        regime_mode = getattr(hmm_model, '_regime_start_mode', 'today_probs')
        K           = getattr(hmm_model, '_smoothing_window', 5)
        decay       = getattr(hmm_model, '_smoothing_decay', 0.3)
        blend_alpha = getattr(hmm_model, '_stationary_blend_alpha', 0.70)
        all_probs   = hmm_model.regime_probs    # (T, n_states)

        K_eff  = min(K, len(all_probs))
        raw_w  = np.exp(-decay * np.arange(K_eff - 1, -1, -1))
        raw_w /= raw_w.sum()
        current_smoothed = np.dot(raw_w, all_probs[-K_eff:])   # probability vector

        eigenvalues, eigvecs = np.linalg.eig(trans_mat.T)
        stat_idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        pi = np.abs(eigvecs[:, stat_idx]); pi /= pi.sum()

        if regime_mode == 'today_probs':
            # TODAY's probability vector (NO argmax — uses the full distribution)
            start_probs   = current_smoothed.copy()
            current_state = int(np.argmax(current_smoothed))

        elif regime_mode == 'blended':
            start_probs   = blend_alpha * pi + (1 - blend_alpha) * current_smoothed
            start_probs  /= start_probs.sum()
            current_state = int(np.argmax(start_probs))

        else:  # 'current' — argmax, legacy
            start_probs   = None
            current_state = int(np.argmax(current_smoothed))

        # ── Current regime logging ───────────────────────────────────────
        raw_state    = int(np.argmax(all_probs[-1]))
        smooth_state = int(np.argmax(current_smoothed))
        raw_label    = MarketRegimeHMM.REGIME_LABELS.get(
            hmm_model.regime_mapping.get(raw_state, 0), '?')
        smooth_label = MarketRegimeHMM.REGIME_LABELS.get(
            hmm_model.regime_mapping.get(smooth_state, 0), '?')

        if regime_mode == 'today_probs':
            dist_str = {MarketRegimeHMM.REGIME_LABELS.get(
                hmm_model.regime_mapping.get(s, s), '?'): f"{current_smoothed[s]*100:.1f}%"
                        for s in range(n_states)}
            print(f"       Mode: FORWARD-LOOKING ({self.seq_length}-day horizon, today={smooth_label})")
            print(f"       Today's distribution (smoothed, K={K_eff}d): {dist_str}")
            if raw_state != smooth_state:
                print(f"       [SMOOTH] Daily: {raw_label} → K={K_eff}: {smooth_label}")
        elif regime_mode == 'blended':
            print(f"       Mode: BLENDED (α={blend_alpha:.2f} stationary + {1-blend_alpha:.2f} today)")
            print(f"       Today's regime: {smooth_label}")
        else:
            print(f"       Mode: CURRENT (regime={smooth_label}, K={K_eff}d)")
            if raw_state != smooth_state:
                print(f"       [SMOOTH] Daily: {raw_label} → Smoothed: {smooth_label}")
        rt_display = getattr(self, 'return_tilt', 0.0)
        if rt_display > 0:
            print(f"\n[INFO] Joint Stochastic Programming ({self.objective}, return_tilt={rt_display:.2f}):")
        else:
            print(f"\n[INFO] Joint Stochastic Programming ({self.objective}):")
        print(f"       Current state: HMM-{current_state} | "
              f"Regime: {MarketRegimeHMM.REGIME_LABELS.get(hmm_model.regime_mapping.get(current_state,0), '?')}")
        print(f"       Simulation: {self.n_sequences} paths x {self.seq_length} days")

        state_counts = self._simulate_sequences(
            trans_mat, current_state, n_states,
            start_probs=start_probs if regime_mode != 'current' else None,
            rng=np.random.default_rng(42)
        )

        mean_counts = state_counts.mean(axis=0)
        for s in range(n_states):
            eco   = hmm_model.regime_mapping.get(s, s)
            label = MarketRegimeHMM.REGIME_LABELS.get(eco, f"R{s}")
            print(f"       {label:20s}: {mean_counts[s]:.1f} expected days out of {self.seq_length}")

        bounds      = [(self.min_weight, self.max_weight)] * n_assets
        constraints = self._build_constraints(ticker_names, sector_map)

        if self.prev_weights is not None and (self.l2_turnover_penalty > 0
                                              or self.transaction_cost_rate > 0):
            w_ref = self.prev_weights
        else:
            w_ref = None

        if self.objective == "max_avg_sharpe":
            obj_fn = lambda w: self._objective_max_avg_sharpe(
                w, mu_matrix, sigma_tensor, state_counts, w_ref)
        elif self.objective == "min_avg_var":
            obj_fn = lambda w: self._objective_min_avg_var(
                w, mu_matrix, sigma_tensor, state_counts, w_ref)
        else:
            obj_fn = lambda w: self._objective_min_avg_cvar(
                w, mu_matrix, sigma_tensor, state_counts, w_ref)

        mu_eff_fallback = np.mean(mu_matrix, axis=0)
        w_feasible = self._get_feasible_start(
            n_assets, ticker_names, sector_map, bounds, mu_eff=mu_eff_fallback)

        candidate_starts = self._candidate_starts(n_assets, mu_matrix, w_feasible)
        n_starts = len(candidate_starts)

        if self.use_nested_cv and self.n_sequences >= 20:
            n_train = int(self.n_sequences * self.nested_cv_ratio)  # 70%
            state_counts_train = state_counts[:n_train]
            state_counts_val   = state_counts[n_train:]
            if self.objective == "max_avg_sharpe":
                obj_fn_train = lambda w: self._objective_max_avg_sharpe(
                    w, mu_matrix, sigma_tensor, state_counts_train, w_ref)
                obj_fn_val   = lambda w: self._objective_max_avg_sharpe(
                    w, mu_matrix, sigma_tensor, state_counts_val, None)
            elif self.objective == "min_avg_var":
                obj_fn_train = lambda w: self._objective_min_avg_var(
                    w, mu_matrix, sigma_tensor, state_counts_train, w_ref)
                obj_fn_val   = lambda w: self._objective_min_avg_var(
                    w, mu_matrix, sigma_tensor, state_counts_val, None)
            else:
                obj_fn_train = lambda w: self._objective_min_avg_cvar(
                    w, mu_matrix, sigma_tensor, state_counts_train, w_ref)
                obj_fn_val   = lambda w: self._objective_min_avg_cvar(
                    w, mu_matrix, sigma_tensor, state_counts_val, None)
        else:
            obj_fn_train = obj_fn
            obj_fn_val   = obj_fn

        candidate_weights = []
        candidate_val_scores = []

        best_weights = w_feasible.copy()

        print(f"       Multi-start optimization ({n_starts})...", end=" ")

        last_error = None
        n_raised = 0
        n_not_converged = 0
        for w_init in candidate_starts:
            try:
                res = minimize(
                    obj_fn_train, w_init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 3000, 'ftol': 1e-11}
                )
                if res.success:
                    w_cand = np.clip(res.x, 0, None)
                    w_cand /= w_cand.sum()
                    val_score = obj_fn_val(w_cand)
                    candidate_weights.append(w_cand)
                    candidate_val_scores.append(val_score)
                else:
                    n_not_converged += 1
            except Exception as e:
                last_error = e
                n_raised += 1
                continue

        if candidate_weights:
            best_idx_cv = int(np.argmin(candidate_val_scores))
            best_weights = candidate_weights[best_idx_cv]

        if candidate_weights:
            print(f"OK ({len(candidate_weights)}/{n_starts} starts converged)")
        else:
            print(f"WARNING — NO START CONVERGED: {n_raised} raised errors (last: {last_error}), "
                  f"{n_not_converged} did not converge.")
            print("       The result below is the sector-feasible STARTING POINT, not an optimized portfolio.")
        best_weights = np.clip(best_weights, 0, None)
        best_weights /= best_weights.sum()

        best_weights = self._enforce_sector_constraints(
            best_weights, n_assets, ticker_names, sector_map, bounds)

        if self.erc_blend > 0.0:
            sigma_avg = np.zeros((n_assets, n_assets))
            for hmm_state in range(n_states):
                eco_state = hmm_to_eco.get(hmm_state, hmm_state)
                if eco_state in regime_stats:
                    p_sim = float(mean_counts[hmm_state]) / self.seq_length
                    sigma_avg += p_sim * regime_stats[eco_state]['sigma']

            try:
                def erc_obj(w):
                    w = np.maximum(w, 1e-8)
                    pv = w @ sigma_avg @ w
                    if pv < 1e-12:
                        return 1e10
                    rc = w * (sigma_avg @ w) / pv
                    return float(np.sum((rc - rc.mean())**2))

                res_erc = minimize(erc_obj, best_weights,
                                   method='SLSQP', bounds=bounds,
                                   constraints=constraints,
                                   options={'maxiter': 1000})
                if res_erc.success:
                    erc_w = np.clip(res_erc.x, 0, None)
                    erc_w /= erc_w.sum()
                    best_weights = (1 - self.erc_blend) * best_weights + self.erc_blend * erc_w
                    best_weights = np.clip(best_weights, 0, None)
                    best_weights /= best_weights.sum()
                    print(f"       ERC blend={self.erc_blend} applied")
            except Exception as e:
                print(f"       [WARN] ERC blend failed: {e}")

        sim_mix = state_counts.mean(axis=0) / self.seq_length  # (n_states,) share of days
        sigma_eff_sim = sum(float(sim_mix[s]) * sigma_tensor[s] for s in range(n_states))
        mu_eff_sim    = sum(float(sim_mix[s]) * mu_matrix[s]    for s in range(n_states))

        p_ret  = float(np.dot(best_weights, mu_eff_sim)) * 252 / self.seq_length
        var_eff = float(np.dot(best_weights, np.dot(sigma_eff_sim, best_weights)))
        p_vol   = float(np.sqrt(max(var_eff * 252 / self.seq_length, 1e-10)))
        sharpe  = (p_ret - self.rf) / p_vol if p_vol > 1e-10 else 0.0

        print(f"       Return={p_ret*100:.2f}% | Vol={p_vol*100:.2f}% | Sharpe={sharpe:.3f}")

        def _eco_label(s):
            return MarketRegimeHMM.REGIME_LABELS.get(hmm_to_eco.get(s, s), f'R{s}')

        labels_map        = {s: _eco_label(s) for s in range(n_states)}
        smoothed_labelled = {_eco_label(s): float(current_smoothed[s]) for s in range(n_states)}
        pi_labelled       = {_eco_label(s): float(pi[s]) for s in range(n_states)}

        report = {
            'mode': f'JSP ({self.objective})',
            'current_state': current_state,
            'n_sequences': self.n_sequences,
            'seq_length': self.seq_length,
            'smoothing_window': K_eff,
            'mean_state_counts': {labels_map[s]: float(mean_counts[s]) for s in range(n_states)},
            'current_probs': hmm_model.get_current_regime_probs(),
            'applied_probs': smoothed_labelled,
            'applied_probs_smoothed': smoothed_labelled,
            'stationary_probs': pi_labelled,
            'regime_probs_full': hmm_model.regime_probs,
            'regime_dates': list(hmm_model.benchmark_index),
            'regime_labels_map': labels_map,
            'regime_portfolios': {},   # JSP does not use per-regime portfolios
            'jsp_sharpe': sharpe,
            'jsp_return': p_ret,
            'jsp_vol': p_vol,
            'n_converged_starts': len(candidate_weights),
            'n_starts': n_starts,
        }

        return best_weights, report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: OUTPUT AND VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioReporter:

    def __init__(self, tickers: List[str], sector_map: Optional[Dict] = None,
                 config: Optional[Dict] = None):
        self.tickers    = tickers
        self.sector_map = sector_map or {}
        self.config     = config or {}

    def print_allocation_report(self, final_weights: np.ndarray,
                                probability_report: Dict,
                                regime_stats: Dict) -> pd.DataFrame:

        print("\n" + "═" * 70)
        print("  OPTIMIZED PORTFOLIO — FINAL ETF ALLOCATION")
        print("═" * 70)

        df_alloc = pd.DataFrame({
            'ETF': self.tickers,
            'Sector': [self.sector_map.get(t, 'N/A') for t in self.tickers],
            'Weight (%)': np.round(final_weights * 100, 2)
        }).sort_values('Weight (%)', ascending=False)

        df_alloc['Weight (%)'] = df_alloc['Weight (%)'].apply(lambda x: f"{x:.2f}%")
        print(df_alloc.to_string(index=False))

        # Summary by sector
        print("\n" + "─" * 40)
        print("  ALLOCATION BY SECTOR:")
        sector_alloc = {}
        for ticker, weight in zip(self.tickers, final_weights):
            sector = self.sector_map.get(ticker, 'N/A')
            sector_alloc[sector] = sector_alloc.get(sector, 0) + weight
        for sector, weight in sorted(sector_alloc.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 50)
            print(f"  {sector:25s}: {weight*100:5.1f}% {bar}")

        print("\n" + "─" * 40)
        w = final_weights

        jsp_ret    = probability_report.get('jsp_return', None)
        jsp_vol    = probability_report.get('jsp_vol', None)
        jsp_sharpe = probability_report.get('jsp_sharpe', None)
        rf         = float(self.config.get('risk_free_rate', 0.03))
        arith      = self.config.get('mu_convention', 'arithmetic') == 'arithmetic'
        n_paths    = probability_report.get('n_sequences', '?')
        horizon    = probability_report.get('seq_length', '?')
        k_window   = probability_report.get('smoothing_window', '?')

        applied_probs = probability_report.get('applied_probs_smoothed',
                        probability_report.get('applied_probs', {}))
        mu_agg    = np.zeros(len(self.tickers))
        sigma_agg = np.zeros((len(self.tickers), len(self.tickers)))
        for eco_state, stats in regime_stats.items():
            label = stats['label']
            prob  = applied_probs.get(label, 0.0)
            mu_agg    += prob * stats['mu']
            sigma_agg += prob * stats['sigma']
        rep_ret    = float(np.dot(w, mu_agg))
        rep_vol    = float(np.sqrt(np.dot(w, np.dot(sigma_agg, w))))
        rep_sharpe = (rep_ret - rf) / rep_vol if rep_vol > 0 else 0.0

        # ── Output ──────────────────────────────────────────────────────────
        print("  PORTFOLIO METRICS — TWO VIEWS:")
        print()
        if jsp_ret is not None:
            print(f"  ① JSP  ({n_paths} simulated paths x {horizon} days, K={k_window}-day smoothing):")
            print(f"     Expected return (p.a.):     {jsp_ret*100:.2f}%")
            print(f"     Expected volatility (p.a.): {jsp_vol*100:.2f}%")
            print(f"     Sharpe ratio (rf={rf*100:.1f}%):      {jsp_sharpe:.3f}")
            if arith:
                print(f"     ≈ compound growth (p.a.):   {(jsp_ret - 0.5 * jsp_vol**2)*100:.2f}%   (return − vol²/2)")
        print()
        print("  ② REPORTER (smoothed regime weights + tilt — context only):")
        print(f"     Expected return (p.a.):     {rep_ret*100:.2f}%")
        print(f"     Expected volatility (p.a.): {rep_vol*100:.2f}%")
        print(f"     Sharpe ratio (rf={rf*100:.1f}%):      {rep_sharpe:.3f}")
        if arith:
            print(f"     ≈ compound growth (p.a.):   {(rep_ret - 0.5 * rep_vol**2)*100:.2f}%   (return − vol²/2)")
        if jsp_ret is not None:
            gap = (rep_ret - jsp_ret)*100
            sign = '+' if gap >= 0 else ''
            print(f"  ↳  Δ Reporter vs JSP:          {sign}{gap:.2f}% "
                  f"({'different regime mix' if abs(gap) > 1 else 'consistent'})")

        print("\n" + "═" * 70)

        return df_alloc

    def plot_results(self, final_weights: np.ndarray,
                     probability_report: Dict,
                     regime_stats: Dict,
                     hmm_model: MarketRegimeHMM,
                     benchmark_returns: pd.Series,
                     save_path: Optional[str] = None) -> None:

        fig = plt.figure(figsize=(22, 20))
        fig.patch.set_facecolor('#0d1117')
        gs = gridspec.GridSpec(3, 3, figure=fig,
                               hspace=0.45, wspace=0.35,
                               height_ratios=[1, 1, 1.15])

        # ── Panel 1: Allocation ETF (Pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_pie(ax1, final_weights)

        # ── Panel 2: Allocation per Sector (Bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sector_bar(ax2, final_weights)

        # ── Panel 3: Regime probabilities (Stacked area chart)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_regime_probs(ax3, probability_report)

        # ── Panel 4: Regime on benchmark (time series)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_regime_timeline(ax4, benchmark_returns, hmm_model)

        # ── Panel 5: Weights per regime (bar chart)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_regime_weights(ax5, probability_report)

        # ── Panel 6: Efficient Frontier (scatter + line + points)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_efficient_frontier(ax6, final_weights, probability_report, regime_stats)

        fig.suptitle(
            'HMM REGIME-SWITCHING PORTFOLIO OPTIMIZER — ETF\n'
            f'Updated: {datetime.today().strftime("%d/%m/%Y")}',
            color='white', fontsize=14, fontweight='bold', y=1.005
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#0d1117')
            print(f"\n[OK] Dashboard saved to: {save_path}")
            frontier_path = save_path.replace('.png', '_frontier.png')
            try:
                fig2, ax_ef = plt.subplots(figsize=(14, 8))
                fig2.patch.set_facecolor('#0d1117')
                self._plot_efficient_frontier(ax_ef, final_weights,
                                              probability_report, regime_stats)
                fig2.suptitle(
                    'Efficient Frontier — JSP Portfolio\n'
                    f'{datetime.today().strftime("%d/%m/%Y")}',
                    color='white', fontsize=13, fontweight='bold'
                )
                fig2.tight_layout()
                fig2.savefig(frontier_path, dpi=180, bbox_inches='tight',
                             facecolor='#0d1117')
                plt.close(fig2)
                print(f"[OK] Efficient frontier saved to: {frontier_path}")
            except Exception as e:
                print(f"[WARN] Separate frontier chart not saved: {e}")

        plt.show()
        plt.close(fig)

    def _frontier_cached(self, key, name, fn):
        """Heavy frontier computations are shared by the dashboard and the stand-alone figure."""
        cache = self.__dict__.setdefault('_frontier_cache', {})
        if (key, name) not in cache:
            cache[(key, name)] = fn()
        return cache[(key, name)]

    @staticmethod
    def _ef_curve(mu, sigma, n_points=60):
        """Long-only minimum-variance frontier: (vols %, returns %)."""
        n = len(mu)
        targets = np.linspace(float(np.mean(mu)) * 0.85, float(np.max(mu)) * 0.98, n_points)
        vols, rets, w_prev = [], [], np.ones(n) / n
        for t in targets:
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w, t=t: float(np.dot(w, mu)) - t}]
            try:
                res = minimize(lambda w: float(np.dot(w, np.dot(sigma, w))), w_prev, method='SLSQP',
                               bounds=[(0.0, 1.0)] * n, constraints=cons,
                               options={'maxiter': 500, 'ftol': 1e-10})
                if res.success:
                    w_f = np.clip(res.x, 0, None); w_f /= w_f.sum()
                    vols.append(float(np.sqrt(max(w_f @ sigma @ w_f, 1e-10))) * 100)
                    rets.append(float(w_f @ mu) * 100)
                    w_prev = w_f
            except Exception:
                pass
        return vols, rets

    @staticmethod
    def _max_sharpe_weights(mu, sigma, rf, bounds, constraints, seed_offset=0, n_starts=12):
        """Best-of-n_starts SLSQP maximum-Sharpe weights (None if no start converges)."""
        n = len(mu)
        lo = min(b[0] for b in bounds); hi = max(b[1] for b in bounds)

        def neg_sharpe(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            if s < 1e-8:
                return 1e10
            w = w / s
            return -(float(w @ mu) - rf) / (float(np.sqrt(max(w @ sigma @ w, 1e-10))) + 1e-10)

        best, best_val = None, np.inf
        for seed in range(n_starts):
            w0 = np.random.default_rng(seed * 31 + seed_offset).dirichlet(np.ones(n) * 0.5)
            w0 = np.clip(w0, lo, hi); w0 /= w0.sum()
            try:
                res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                               options={'maxiter': 800, 'ftol': 1e-11})
                if res.success and res.fun < best_val:
                    best_val, best = res.fun, res.x.copy()
            except Exception:
                pass
        if best is None:
            return None
        best = np.clip(best, 0, None)
        return best / best.sum()

    def _plot_efficient_frontier(self, ax, final_weights, probability_report, regime_stats):
        ax.set_facecolor('#161b22')
        ax.spines[:].set_color('#30363d')
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xlabel('Annualized volatility (%)', color='white', fontsize=10)
        ax.set_ylabel('Expected annualized return (%)', color='white', fontsize=10)
        ax.set_title('Efficient Frontier — Simulated & Optimal Portfolios',
                     color='white', fontsize=12, fontweight='bold', pad=12)
        ax.grid(color='#30363d', alpha=0.4, linewidth=0.5)

        try:
            rf = float(self.config.get('risk_free_rate', 0.03))
            mean_counts_dict = (probability_report.get('mean_state_counts')
                                or probability_report.get('applied_probs', {}))
            eco_labels_avail = [s for s in regime_stats]
            n_assets = len(self.tickers)
            # regime_stats is keyed by economic-state id, mean_state_counts by label
            label_of = lambda k: MarketRegimeHMM.REGIME_LABELS.get(k, k)

            mu_by_regime    = {s: regime_stats[s]['mu']    for s in eco_labels_avail}
            sigma_by_regime = {s: regime_stats[s]['sigma'] for s in eco_labels_avail}

            total_days = sum(mean_counts_dict.get(label_of(s), 0) for s in eco_labels_avail)
            if total_days < 1e-6:
                print("  [WARN] Frontier panel: simulated regime mix not found — using an equal mix.")
                mix = {s: 1.0 / len(eco_labels_avail) for s in eco_labels_avail}
            else:
                mix = {s: mean_counts_dict.get(label_of(s), 0) / total_days for s in eco_labels_avail}

            mu_eff    = sum(mix[s] * mu_by_regime[s]    for s in eco_labels_avail)
            sigma_eff = sum(mix[s] * sigma_by_regime[s] for s in eco_labels_avail)

            mu_ann    = mu_eff      # already p.a. — no rescaling
            sigma_ann = sigma_eff   # already p.a. — no rescaling

            def port_stats(w):
                """Annualized return and volatility for weights w."""
                ret = float(np.dot(w, mu_ann))
                var = float(np.dot(w, np.dot(sigma_ann, w)))
                vol = float(np.sqrt(max(var, 1e-10)))
                return ret, vol

            # ── 1. Monte Carlo: 3000 random portfolios ──────────────────
            n_mc  = 3000
            W_mc = np.random.default_rng(42).dirichlet(np.ones(n_assets) * 0.5, size=n_mc)
            mc_r = W_mc @ mu_ann
            mc_vols    = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', W_mc, sigma_ann, W_mc), 1e-10))
            mc_sharpes = (mc_r - rf) / (mc_vols + 1e-10)
            mc_rets, mc_vols = mc_r * 100, mc_vols * 100

            # Colormap: low Sharpe (red) → high Sharpe (green)
            scatter = ax.scatter(
                mc_vols, mc_rets,
                c=mc_sharpes, cmap='RdYlGn',
                alpha=0.35, s=8, zorder=2,
                vmin=np.percentile(mc_sharpes, 5),
                vmax=np.percentile(mc_sharpes, 95)
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.025)
            cbar.set_label('Sharpe Ratio', color='white', fontsize=8)
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

            fkey = (id(probability_report), id(regime_stats))
            frontier_vols, frontier_rets = self._frontier_cached(
                fkey, 'curve', lambda: self._ef_curve(mu_ann, sigma_ann))

            if len(frontier_vols) >= 3:
                ef_sorted = sorted(zip(frontier_vols, frontier_rets))
                ef_v = [x[0] for x in ef_sorted]
                ef_r = [x[1] for x in ef_sorted]
                ax.plot(ef_v, ef_r,
                        color='#58a6ff', linewidth=2.2,
                        zorder=4, label='Efficient Frontier')

            # ── 3. MAX SHARPE point ──────────────
            best_ms = self._frontier_cached(fkey, 'ms', lambda: self._max_sharpe_weights(
                mu_ann, sigma_ann, rf, [(0.0, 1.0)] * n_assets, _slsqp_constraints(self.tickers, None, None)))

            if best_ms is not None:
                r_ms, v_ms = port_stats(best_ms)
                sharpe_ms  = (r_ms - rf) / (v_ms + 1e-10)
                ax.scatter(
                    v_ms * 100, r_ms * 100,
                    marker='*', s=420, color='#FFD700',
                    zorder=6, edgecolors='white', linewidths=0.8,
                    label=f'Max Sharpe (theoretical): SR={sharpe_ms:.2f}  (r={r_ms*100:.1f}%, σ={v_ms*100:.1f}%)'
                )
                v_range = np.linspace(0, max(mc_vols) * 1.05, 100)
                r_cml   = rf * 100 + sharpe_ms * v_range
                ax.plot(v_range, r_cml,
                        color='#FFD700', linewidth=1.0,
                        linestyle='--', alpha=0.55, zorder=3,
                        label='CML (Capital Market Line)')

            sector_map_cfg    = self.config.get('sector_map', {})
            sector_constr_cfg = self.config.get('sector_constraints', {})
            min_w_cfg = self.config.get('min_weight', 0.0)
            max_w_cfg = self.config.get('max_weight', 1.0)
            best_ms_c = self._frontier_cached(fkey, 'msc', lambda: self._max_sharpe_weights(
                mu_ann, sigma_ann, rf, [(min_w_cfg, max_w_cfg)] * n_assets,
                _slsqp_constraints(self.tickers, sector_map_cfg, sector_constr_cfg), seed_offset=7))

            if best_ms_c is not None:
                r_msc, v_msc = port_stats(best_ms_c)
                sharpe_msc = (r_msc - rf) / (v_msc + 1e-10)
                has_constraints = bool(sector_constr_cfg) or min_w_cfg > 0
                label_c = (f'Max Sharpe (constrained): SR={sharpe_msc:.2f}  (r={r_msc*100:.1f}%, σ={v_msc*100:.1f}%)' if has_constraints
                           else f'Max Sharpe (no tilt): SR={sharpe_msc:.2f}  (r={r_msc*100:.1f}%, σ={v_msc*100:.1f}%)')
                ax.scatter(
                    v_msc * 100, r_msc * 100,
                    marker='*', s=420, color='#2ea043',
                    zorder=6, edgecolors='white', linewidths=0.8,
                    label=label_c
                )

            r_jsp = probability_report.get('jsp_return', None)
            v_jsp = probability_report.get('jsp_vol', None)
            sharpe_jsp = probability_report.get('jsp_sharpe', None)

            star_name = 'JSP Portfolio'
            if r_jsp is None or v_jsp is None:          # classic modes: evaluate the final weights
                r_jsp, v_jsp = port_stats(np.asarray(final_weights))
                sharpe_jsp = (r_jsp - rf) / (v_jsp + 1e-10)
                star_name = 'Final portfolio'
            ax.scatter(
                v_jsp * 100, r_jsp * 100,
                marker='*', s=500, color='#00E5FF',
                zorder=7, edgecolors='white', linewidths=0.8,
                label=f'{star_name}: SR={sharpe_jsp:.2f}  (r={r_jsp*100:.1f}%, σ={v_jsp*100:.1f}%)'
            )

            # ──  risk-free ────────────────────────────────────────────
            ax.axhline(rf * 100, color='#8b949e', linewidth=0.8,
                       linestyle=':', alpha=0.7, zorder=2,
                       label=f'Risk-free ({rf*100:.1f}%)')

            # ── Legend and layout ───────────────────────────────────────────
            ax.legend(
                loc='upper left', fontsize=8.5, labelcolor='white',
                facecolor='#1c2128', edgecolor='#30363d',
                framealpha=0.9
            )

            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#30363d')

            note = ("Monte Carlo: 3000 random portfolios (box [0,1]). "
                    "Efficient frontier: min-vol for each target-return. "
                    "Theoretical Max Sharpe: no sector constraints. "
                    "Star: final portfolio, with sector constraints.")
            ax.text(0.01, -0.09, note,
                    transform=ax.transAxes,
                    color='#8b949e', fontsize=7, style='italic')

        except Exception as e:
            ax.text(0.5, 0.5,
                    f'Frontier not available\n({e})',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#8b949e', fontsize=10)

    def _plot_pie(self, ax, weights):
        ax.set_facecolor('#161b22')
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        threshold = 0.04
        labels = [t if w >= threshold else '' for t, w in zip(self.tickers, weights)]
        _, texts, autotexts = ax.pie(
            weights, labels=labels, autopct=lambda p: f'{p:.1f}%' if p >= 4 else '',
            colors=colors, startangle=90, pctdistance=0.8
        )
        for t in texts + autotexts:
            t.set_color('white')
            t.set_fontsize(7)
        ax.set_title('ETF Allocation', color='white', fontsize=11, pad=10)
        n_small = int(np.sum(np.asarray(weights) < threshold))
        if n_small:
            ax.text(0.5, -0.02, f'{n_small} ETFs below {threshold*100:.0f}% not labelled', transform=ax.transAxes,
                    ha='center', color='#8b949e', fontsize=7)

    def _plot_sector_bar(self, ax, weights):
        ax.set_facecolor('#161b22')
        sector_alloc = {}
        for ticker, w in zip(self.tickers, weights):
            sector = self.sector_map.get(ticker, 'N/A')
            sector_alloc[sector] = sector_alloc.get(sector, 0) + w

        sectors = sorted(sector_alloc, key=sector_alloc.get)      # largest sector on top
        vals = [sector_alloc[s] * 100 for s in sectors]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sectors)))

        bars = ax.barh(sectors, vals, color=colors, edgecolor='#30363d')
        ax.set_xlabel('Weight (%)', color='white')
        ax.set_title('Allocation by Sector', color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#30363d')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', color='white', va='center', fontsize=8)

    def _plot_regime_probs(self, ax, probability_report):

        ax.set_facecolor('#161b22')

        regime_probs_full = probability_report.get('regime_probs_full', None)
        regime_dates      = probability_report.get('regime_dates', None)
        regime_labels_map = probability_report.get('regime_labels_map', {})

        if regime_probs_full is None or regime_dates is None:
            probs  = probability_report.get('applied_probs', {})
            labels = list(probs.keys())
            x      = np.arange(len(labels))
            colors_map = {'Bull':'#2ea043','Sideways/Kangaroo':'#e3b341','Bear':'#f85149'}
            bar_c  = [colors_map.get(l,'#58a6ff') for l in labels]
            ax.bar(x, [probs.get(l,0)*100 for l in labels], 0.6, color=bar_c)
            ax.set_xticks(x)
            ax.set_xticklabels([l.split('/')[0] for l in labels], color='white', fontsize=8)
            ax.set_title('Current Regime Probabilities', color='white', fontsize=10)
            ax.tick_params(colors='white'); ax.spines[:].set_color('#30363d')
            return

        colors_map = {'Bull':'#2ea043','Sideways/Kangaroo':'#e3b341',
                      'Bear':'#f85149','Sideways':'#e3b341'}
        dates_arr = np.array(regime_dates)
        n_states  = regime_probs_full.shape[1]

        smooth = pd.DataFrame(regime_probs_full).rolling(21, min_periods=1).mean().values   # display only
        order = sorted(range(n_states), key=lambda s: regime_labels_map.get(s, 'Z'))
        ax.stackplot(dates_arr, *[smooth[:, s] * 100 for s in order],
                     colors=[colors_map.get(regime_labels_map.get(s, ''), '#58a6ff') for s in order],
                     labels=[regime_labels_map.get(s, f'R{s}').split('/')[0] for s in order],
                     alpha=0.85)

        ax.axvline(dates_arr[-1], color='white', linewidth=1.5,
                   linestyle='--', alpha=0.8, label='Today')

        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)', color='white', fontsize=8)
        ax.set_title('Regime Probabilities (21-day avg)', color='white',
                     fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, labelcolor='white',
                  facecolor='#161b22', edgecolor='#30363d')
        ax.tick_params(colors='white', labelsize=7)
        ax.spines[:].set_color('#30363d')
        ax.grid(color='#30363d', alpha=0.3, linewidth=0.5)
        try:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        except Exception:
            pass

    def _plot_regime_timeline(self, ax, benchmark_returns, hmm_model):
        ax.set_facecolor('#161b22')

        cum_returns = np.exp(benchmark_returns.cumsum())   # log returns -> cumulative growth
        dates = benchmark_returns.index[-len(hmm_model.regime_sequence):]

        ax.plot(dates, cum_returns.values[-len(dates):],
                color='#58a6ff', linewidth=1.2, label='Benchmark', zorder=3)

        regime_colors = {0: '#e3b341', 1: '#2ea043', 2: '#f85149'}
        prev_regime = None
        start_idx = 0

        for i, (date, regime_hmm) in enumerate(zip(dates, hmm_model.regime_sequence)):
            economic_state = hmm_model.regime_mapping.get(regime_hmm, regime_hmm)
            if economic_state != prev_regime:
                if prev_regime is not None and i > 0:
                    ax.axvspan(dates[start_idx], date,
                               alpha=0.15,
                               color=regime_colors.get(prev_regime, 'gray'),
                               zorder=1)
                prev_regime = economic_state
                start_idx = i

        if prev_regime is not None:
            ax.axvspan(dates[start_idx], dates[-1],
                       alpha=0.15, color=regime_colors.get(prev_regime, 'gray'), zorder=1)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ea043', alpha=0.5, label='Bull'),
            Patch(facecolor='#e3b341', alpha=0.5, label='Sideways'),
            Patch(facecolor='#f85149', alpha=0.5, label='Bear')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  labelcolor='white', facecolor='#161b22', edgecolor='#30363d')
        ax.set_title('Market Regimes on the Benchmark (time series)', color='white', fontsize=11)
        ax.set_ylabel('Cumulative Return', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#30363d')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    def _plot_regime_weights(self, ax, probability_report):
        ax.set_facecolor('#161b22')
        regime_portfolios = probability_report.get('regime_portfolios', {})
        applied_probs = probability_report['applied_probs']

        if not regime_portfolios:
            mix = probability_report.get('mean_state_counts', {})
            total = float(sum(mix.values()))
            if total > 0:
                palette = {'Bull': '#2ea043', 'Sideways/Kangaroo': '#e3b341', 'Bear': '#f85149'}
                names = list(mix.keys())
                shares = [100.0 * mix[k] / total for k in names]
                bars = ax.bar(names, shares, color=[palette.get(k, 'gray') for k in names], alpha=0.85)
                for b, v in zip(bars, shares):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.0f}%',
                            ha='center', color='white', fontsize=9)
                ax.set_ylim(0, max(shares) * 1.25)
                ax.set_title('Expected Regime Mix (simulated paths)', color='white', fontsize=11)
                ax.set_ylabel('Share of horizon (%)', color='white')
                ax.tick_params(colors='white', labelsize=8)
                ax.spines[:].set_color('#30363d')
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        color='white', ha='center', va='center')
            return

        x = np.arange(len(self.tickers))
        width = 0.8 / max(len(regime_portfolios), 1)

        for i, (regime_label, port_w) in enumerate(regime_portfolios.items()):
            if port_w is None:
                continue
            prob = applied_probs.get(regime_label, 0)
            offset = (i - len(regime_portfolios) / 2) * width
            ax.bar(x + offset, port_w * 100, width,
                   label=f"{regime_label.split('/')[0]} ({prob*100:.0f}%)",
                   alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(self.tickers, rotation=45, ha='right', color='white', fontsize=6)
        ax.set_ylabel('Weight (%)', color='white')
        ax.set_title('Weights by Regime', color='white', fontsize=11)
        ax.legend(fontsize=7, labelcolor='white', facecolor='#161b22', edgecolor='#30363d')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#30363d')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def validate_config(config: Dict) -> None:
    """Fail fast on silent configuration mismatches and infeasible constraints."""
    tickers = list(config['etf_tickers'])
    smap = config.get('sector_map', {}) or {}
    priors = config.get('sector_priors', {}) or {}
    cons = config.get('sector_constraints', {}) or {}
    min_w = float(config.get('min_weight', 0.0))
    max_w = float(config.get('max_weight', 1.0))
    errors, warns = [], []

    dup = sorted({t for t in tickers if tickers.count(t) > 1})
    if dup:
        errors.append(f"duplicate tickers: {dup}")
    missing_map = [t for t in tickers if t not in smap]
    if missing_map:
        warns.append(f"tickers without a sector in sector_map: {missing_map}")
    extra_map = [t for t in smap if t not in tickers]
    if extra_map:
        warns.append(f"sector_map entries not in etf_tickers: {extra_map}")

    sectors = sorted({smap[t] for t in tickers if t in smap})
    no_prior = [s for s in sectors if s not in priors]
    if no_prior:
        warns.append(f"sectors without a prior (fall back to DEFAULT): {no_prior}")
    unknown_cons = [s for s in cons if s not in sectors]
    if unknown_cons:
        warns.append(f"sector_constraints for sectors with no ETF: {unknown_cons}")

    n = len(tickers)
    if n * min_w > 1 + 1e-9:
        errors.append(f"min_weight={min_w} x {n} ETFs exceeds 100%")
    if n * max_w < 1 - 1e-9:
        errors.append(f"max_weight={max_w} x {n} ETFs cannot reach 100%")

    lo_total, hi_total = 0.0, 0.0
    for s in sectors:
        k = sum(1 for t in tickers if smap.get(t) == s)
        s_min, s_max = cons.get(s, (0.0, 1.0))
        lo, hi = max(s_min, k * min_w), min(s_max, k * max_w)
        if lo > hi + 1e-9:
            errors.append(f"sector '{s}' infeasible: effective bounds [{lo:.3f}, {hi:.3f}]")
        lo_total += lo
        hi_total += hi
    if lo_total > 1 + 1e-9:
        errors.append(f"sum of effective sector minimums = {lo_total:.3f} > 1")
    if hi_total < 1 - 1e-9:
        errors.append(f"sum of effective sector maximums = {hi_total:.3f} < 1")

    cw = config.get('current_weights')
    if cw:
        unknown = [t for t in cw if t not in tickers]
        if unknown:
            errors.append(f"current_weights contains tickers not in etf_tickers: {unknown}")
        if any(float(v) < 0 for v in cw.values()):
            errors.append("current_weights must be non-negative (long-only)")
        if abs(sum(float(v) for v in cw.values()) - 1.0) > 0.02:
            warns.append("current_weights do not sum to 1 — they will be normalised")

    rt = float(config.get('return_tilt', 0.0))
    if not 0.0 <= rt <= 1.0:
        errors.append(f"return_tilt={rt} outside [0, 1]")

    allowed = {
        'weighting_mode': {'jsp', 'jsp_var', 'jsp_cvar', 'blend', 'current',
                           'stationary', 'custom', 'market_analysis'},
        'objective_mode': {'max_sharpe', 'blended', 'max_return', 'vol_target'},
        'regime_start_mode': {'today_probs', 'blended', 'current'},
        'wfo_protocol': {'rolling', 'expanding'},
        'mu_convention': {'arithmetic', 'log'},
    }
    for key, valid in allowed.items():
        val = config.get(key)
        if val is not None and val not in valid:
            errors.append(f"{key}='{val}' is not one of {sorted(valid)}")
    if config.get('weighting_mode') == 'custom' and not config.get('custom_regime_weights'):
        errors.append("weighting_mode='custom' requires custom_regime_weights")
    if config.get('objective_mode') == 'vol_target' and config.get('vol_target') is None:
        errors.append("objective_mode='vol_target' requires vol_target")
    floor, n_reg = float(config.get('regime_prob_floor', 0.05)), int(config.get('n_regimes', 3))
    if not 0.0 <= floor < 1.0 / max(n_reg, 1):
        errors.append(f"regime_prob_floor={floor} must be in [0, 1/n_regimes={1.0/max(n_reg, 1):.3f})")

    suffixes = {t.rsplit('.', 1)[-1] for t in tickers if '.' in t}
    if 'L' in suffixes and suffixes & {'AS', 'DE', 'MI', 'PA', 'F', 'MC', 'BR'}:
        warns.append("tickers from London (.L: quoted in GBP, GBX or USD depending on the line) are mixed "
                     "with euro-listed ETFs; returns are NOT converted to a common currency, so FX moves "
                     "end up inside returns and correlations — check the quote currency of each ticker")

    for w in warns:
        print(f"[CONFIG WARN] {w}")
    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    print(f"[OK] Configuration validated ({n} ETFs, {len(sectors)} sectors)")


class HMMPortfolioOptimizer:

    def __init__(self, config: Dict):

        self.config = config
        self.tickers = config['etf_tickers']
        self.sector_map = config.get('sector_map', {})
        self.returns = None
        self.hmm = None
        self.regime_stats = None
        self.regime_portfolios = {}
        self.final_weights = None
        self.probability_report = None

    def run(self) -> pd.DataFrame:

        print("\n" + "█" * 70)
        print("  HMM REGIME-SWITCHING PORTFOLIO OPTIMIZER")
        print("  ETF Portfolio — " + datetime.today().strftime("%d/%m/%Y %H:%M"))
        print("█" * 70)

        validate_config(self.config)

        # ─── Step 1: DATA LOADING ────────────────────────────────────────
        print("\n[STEP 1/6] Loading ETF price histories...")
        loader = DataLoader(
            excel_path=self.config.get('excel_file_path'),
            excel_sheet=self.config.get('excel_sheet', 'Sheet1'),
            excel_date_col=self.config.get('excel_date_col', 'Date')
        )

        if self.config.get('excel_file_path'):
            self.returns = loader.load_from_excel(self.tickers, period_years=self.config.get('period_years', 5))
        else:
            self.returns = loader.load_from_yfinance(
                self.tickers,
                period_years=self.config.get('period_years', 5)
            )

        # Keep tickers, returns columns and sector map consistent if Yahoo dropped any ETF
        available = [t for t in self.tickers if t in self.returns.columns]
        if len(available) < len(self.tickers):
            dropped = [t for t in self.tickers if t not in available]
            print(f"[WARN] ETFs without data are removed from the optimization: {dropped}")
            self.tickers = available
        self.returns = self.returns[self.tickers]

        # ─── Step 2:  benchmark definition ───────────────────────────────────
        print("\n[STEP 2/6] Preparing benchmark for regime detection...")
        benchmark_ticker = self.config.get('benchmark_ticker', self.tickers[0])
        benchmark_period = self.config.get('benchmark_period_years', None)

        if benchmark_period and benchmark_period > self.config.get('period_years', 5):
            print(f"  Benchmark: {benchmark_ticker} with extended history "
                  f"({benchmark_period} years vs {self.config.get('period_years',5)} years for ETFs)")
            try:
                bench_loader = DataLoader()
                bench_long   = bench_loader.load_from_yfinance(
                    [benchmark_ticker],
                    period_years=benchmark_period
                )
                benchmark_returns = bench_long[benchmark_ticker]
                print(f"  [OK] Long benchmark: {len(benchmark_returns)} days "
                      f"({benchmark_returns.index[0].strftime('%d/%m/%Y')} → "
                      f"{benchmark_returns.index[-1].strftime('%d/%m/%Y')})")
            except Exception as e:
                print(f"  [WARN] Long benchmark download failed ({e}) — using short data")
                benchmark_returns = self.returns.get(benchmark_ticker,
                                                      self.returns.mean(axis=1))
        elif benchmark_ticker in self.returns.columns:
            benchmark_returns = self.returns[benchmark_ticker]
            print(f"  Benchmark: {benchmark_ticker} ({len(benchmark_returns)} days)")
        else:
            benchmark_returns = self.returns.mean(axis=1)
            print("  Benchmark: ETF portfolio average")

        # ─── Step 3: Training HMM ─────────────────────────────────────────
        print("\n[STEP 3/6] Training Hidden Markov Model...")
        self.hmm = MarketRegimeHMM(
            n_regimes=self.config.get('n_regimes', 3),
            n_iter=self.config.get('hmm_iterations', 150),
            window_volatility=self.config.get('volatility_window', 10),
            random_state=self.config.get('random_state', 42),
            regime_prob_floor=self.config.get('regime_prob_floor', 0.05)
        )
        self.hmm._sector_map        = self.sector_map
        self.hmm._sector_priors     = self.config.get('sector_priors', None)
        self.hmm._regime_mu_floors  = self.config.get('regime_mu_floors', {})
        self.hmm._prior_schedule    = self.config.get('prior_schedule', None)
        self.hmm._strict_prior_schedule = self.config.get('strict_prior_schedule', False)
        self.hmm._use_mp_denoising  = self.config.get('use_mp_denoising', True)
        self.hmm._lw_preserve_var   = bool(self.config.get('lw_preserve_variances', True))
        self.hmm._mu_convention     = str(self.config.get('mu_convention', 'arithmetic'))
        self.hmm._risk_free         = float(self.config.get('risk_free_rate', 0.03))
        self.hmm._smoothing_window        = self.config.get('regime_smoothing_window', 5)
        self.hmm._smoothing_decay         = self.config.get('regime_smoothing_decay', 0.3)
        self.hmm._regime_start_mode       = self.config.get('regime_start_mode', 'today_probs')
        self.hmm._stationary_blend_alpha  = self.config.get('stationary_blend_alpha', 0.70)
        self.hmm.fit(
            benchmark_returns,
            auto_select_regimes=self.config.get('auto_select_regimes', False),
            n_ensemble=self.config.get('hmm_n_ensemble', 10)
        )
        self.hmm.print_transition_matrix()

        # ─── VALIDATION HMM (Train/Test + Silhouette) ────────────────────
        if self.config.get('run_hmm_validation', True):
            self.hmm._run_validation_after_fit(benchmark_returns)

        # ─── Step 4: FORECASTING (μ, Σ) per regime ─────────────────────────────
        print("\n[STEP 4/6] Estimating (μ, Σ) per regime with Ledoit-Wolf shrinkage...")
        self.regime_stats = self.hmm.get_regime_statistics(self.returns)

        # ── Momentum Tilt (optional) ──────────────────────────────────────
        momentum_tilt = self.config.get('momentum_tilt', 0.0)
        if momentum_tilt > 0:
            print(f"\n[INFO] Applying Momentum Tilt (strength={momentum_tilt})...")
            self.regime_stats = self.hmm.apply_momentum_tilt(
                etf_returns=self.returns,
                mu_by_regime=self.regime_stats,
                momentum_window_long=self.config.get('momentum_window_long', 252),
                momentum_window_short=self.config.get('momentum_window_short', 21),
                tilt_strength=momentum_tilt
            )

        # ─── Step 5+6: OPTIMIZATION AND FINAL WEIGHTING ────────────────────
        weighting_mode = self.config.get('weighting_mode', 'blend')
        objective_mode = self.config.get('objective_mode', 'blended')
        return_tilt_cfg = float(self.config.get('return_tilt', 0.0))
        vol_target_cfg = self.config.get('vol_target', None)
        if objective_mode == "max_sharpe":
            if return_tilt_cfg > 0.0:
                print(f"  [INFO] objective_mode='max_sharpe': return_tilt={return_tilt_cfg:.2f} "
                      f"ignored. Use objective_mode='blended' to enable the tilt.")
            return_tilt_cfg = 0.0
            vol_target_cfg = None
        elif objective_mode == "max_return":
            return_tilt_cfg = 1.0
            vol_target_cfg = None
        elif objective_mode == "vol_target":
            return_tilt_cfg = 0.0
        elif objective_mode == "blended":
            pass

        if weighting_mode in ("jsp", "jsp_var", "jsp_cvar"):
            if weighting_mode == "jsp_var":
                jsp_objective = "min_avg_var"
            elif weighting_mode == "jsp_cvar":
                jsp_objective = "min_avg_cvar"
            else:
                jsp_objective = "max_avg_sharpe"
            print(f"\n[STEP 5/6] Joint Stochastic Programming ({jsp_objective})...")
            # Optional current portfolio -> activates turnover cost / L2 penalty (otherwise they cannot apply)
            current_w = None
            cw = self.config.get('current_weights')
            if cw:
                vec = np.array([float(cw.get(t, 0.0)) for t in self.tickers])
                if vec.sum() > 0:
                    current_w = vec / vec.sum()
            print("\n[STEP 6/6] (included in the JSP — no separate weighting)")

            jsp = JointStochasticOptimizer(
                n_sequences=self.config.get('jsp_n_sequences', 1000),
                seq_length=self.config.get('jsp_seq_length', 63),
                objective=jsp_objective,
                risk_free_rate=self.config.get('risk_free_rate', 0.03),
                min_weight=self.config.get('min_weight', 0.02),
                max_weight=self.config.get('max_weight', 0.40),
                sector_constraints=self.config.get('sector_constraints', {}),
                erc_blend=self.config.get('erc_blend', 0.0),
                var_confidence=self.config.get('var_confidence', 0.05),
                use_nested_cv=self.config.get('use_nested_cv', True),
                nested_cv_ratio=self.config.get('nested_cv_ratio', 0.70),
                l2_turnover_penalty=self.config.get('l2_turnover_penalty', 0.10),
                prev_weights=current_w,
                return_tilt=return_tilt_cfg,
                transaction_cost_bps=self.config.get('transaction_cost_bps', 5.0),
                cvar_alpha=self.config.get('cvar_alpha', 0.05)
            )
            self.final_weights, self.probability_report = jsp.optimize(
                regime_stats=self.regime_stats,
                hmm_model=self.hmm,
                ticker_names=self.tickers,
                sector_map=self.sector_map
            )
        else:
            print("\n[STEP 5/6] Mean-Variance optimization for each regime...")
            optimizer = RegimeMVOptimizer(
                risk_free_rate=self.config.get('risk_free_rate', 0.03),
                min_weight=self.config.get('min_weight', 0.02),
                max_weight=self.config.get('max_weight', 0.40),
                sector_constraints=self.config.get('sector_constraints', {}),
                erc_blend=self.config.get('erc_blend', 0.0)
            )

            return_tilt = return_tilt_cfg
            vol_target  = vol_target_cfg

            for eco_state, stats in self.regime_stats.items():
                weights = optimizer.optimize_for_regime(
                    mu=stats['mu'],
                    sigma=stats['sigma'],
                    n_assets=len(self.tickers),
                    ticker_names=self.tickers,
                    regime_label=stats['label'],
                    sector_map=self.sector_map,
                    return_tilt=return_tilt,
                    vol_target=vol_target
                )
                self.regime_portfolios[stats['label']] = weights

            print("\n[STEP 6/6] Computing the final regime-weighted portfolio...")
            weighter = RegimeProbabilityWeighter(
                weighting_mode=weighting_mode,
                blend_alpha=self.config.get('blend_alpha', 0.4),
                custom_weights=self.config.get('custom_regime_weights')
            )
            self.final_weights, self.probability_report = weighter.compute_final_weights(
                self.regime_portfolios, self.hmm
            )

        reporter = PortfolioReporter(self.tickers, self.sector_map, config=self.config)
        reporter.print_allocation_report(
            self.final_weights, self.probability_report, self.regime_stats
        )

        if self.config.get('show_plots', True):
            try:
                reporter.plot_results(
                    self.final_weights,
                    self.probability_report,
                    self.regime_stats,
                    self.hmm,
                    benchmark_returns,
                    save_path=self.config.get('save_plot_path', 'portfolio_dashboard.png')
                )
            except Exception as e:
                print(f"[WARN] Unable to generate plots: {e}")

        # ─── Walk-Forward Validation (optional) ────────────────────────────────
        if self.config.get('run_walkforward', False) and len(self.returns) >= 252:
            wfv = WalkForwardValidator(
                returns=self.returns,
                config=self.config,
                train_months=self.config.get('wfo_train_months', 18),
                rebal_months=self.config.get('wfo_rebal_months', 3),
                benchmark_col=self.tickers[0]
            )
            self.wfo_results = wfv.run()

        try:
            if len(self.returns) >= 252:
                bench_name = self.tickers[0]
                last_year = np.expm1(self.returns.iloc[-252:])   # log -> simple returns
                port_1y  = float((1 + (last_year * self.final_weights).sum(axis=1)).prod() - 1)
                bench_1y = float((1 + last_year[bench_name]).prod() - 1)
                print(f"\n{'='*60}")
                print("  CONTEXT: HISTORICAL return over the last 12 months")
                print(f"{'='*60}")
                print(f"  Optimized portfolio (naive, current weights): {port_1y*100:+.2f}%")
                print(f"  Benchmark {bench_name:<12s}:                  {bench_1y*100:+.2f}%")
                print(f"  JSP (forward-looking, annualized):            "
                      f"{self.probability_report.get('jsp_return', 0)*100:+.2f}% p.a.")
                print("")
                print("  ⚠ The historical figure contains LOOK-AHEAD BIAS")
                print("    (weights computed today and applied to past data).")
                print("    The JSP figure is the appropriate forward-looking estimate.")
                print(f"{'='*60}")
        except Exception as e:
            print(f"[WARN] 12-month context not computed: {e}")

        output_csv = self.config.get('output_csv', 'portfolio_allocation.csv')
        df_clean = pd.DataFrame({
            'Ticker': self.tickers,
            'Sector': [self.sector_map.get(t, 'N/A') for t in self.tickers],
            'Weight_Pct': np.round(self.final_weights * 100, 4)
        }).sort_values('Weight_Pct', ascending=False)
        df_clean.to_csv(output_csv, index=False)
        print(f"\n[OK] Allocation saved to: {output_csv}")

        return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WALK-FORWARD BACKTEST (validation OOS)
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:

    def __init__(self, returns: pd.DataFrame, config: dict,
                 train_months: int = 24,
                 rebal_months: int = 3,
                 benchmark_col: str = None):
        self.returns     = returns
        self.config      = config
        self.train_months = train_months
        self.rebal_months = rebal_months
        self.benchmark   = benchmark_col or returns.columns[0]
        self.train_days  = train_months * 21
        self.rebal_days  = rebal_months * 21

    def run(self) -> dict:
        ret = self.returns
        n   = len(ret)
        oos_gap_days = int(self.config.get("oos_gap_days", 5))
        wfo_protocol = self.config.get("wfo_protocol", "rolling")
        tx_bps = float(self.config.get("transaction_cost_bps", 5.0))
        tx_rate = max(0.0, tx_bps) / 10000.0
        rf_wfo = float(self.config.get('risk_free_rate', 0.03))

        if n < self.train_days + self.rebal_days:
            print(f"  [WFO WARN] Insufficient data for WFO "
                  f"({n} < {self.train_days + self.rebal_days})")
            return {}

        print(f"\n[WFO] Walk-Forward Backtest ({self.train_months}m train, "
              f"{self.rebal_months}m rebal)...")
        print("      NOTE: proxy backtest — each window re-estimates a James-Stein max-Sharpe")
        print("      portfolio; the HMM regimes and the JSP are NOT re-fitted per window.")

        port_returns = []
        bench_returns = []
        windows = []
        all_weights = []

        wfo_start_date_str = self.config.get('wfo_start_date', None)
        if wfo_start_date_str:
            try:
                wfo_start_ts = pd.Timestamp(wfo_start_date_str)
                idx_candidates = [i for i, d in enumerate(ret.index) if d >= wfo_start_ts]
                start_idx = idx_candidates[0] if idx_candidates else self.train_days
                start_idx = max(start_idx, self.train_days)  # not before the minimum training period
            except Exception as e:
                print(f"  [WFO WARN] wfo_start_date='{wfo_start_date_str}' not usable ({e}) — using the default start")
                start_idx = self.train_days
        else:
            start_idx = self.train_days
        prev_weights = None

        while start_idx + self.rebal_days <= n:
            train_end = start_idx - oos_gap_days
            if train_end <= 0:
                start_idx += self.rebal_days
                continue
            if wfo_protocol == "expanding":
                train_start_idx = 0
            else:
                train_start_idx = max(0, train_end - self.train_days)
            train_data = ret.iloc[train_start_idx:train_end]
            test_data  = ret.iloc[start_idx:start_idx + self.rebal_days]
            if len(train_data) < self.train_days:
                start_idx += self.rebal_days
                continue

            train_start = train_data.index[0].strftime('%Y-%m')
            test_end    = test_data.index[-1].strftime('%Y-%m')

            try:
                cfg = dict(self.config)
                cfg['show_plots'] = False

                sigma_train = _ledoit_wolf_cov(train_data.values, self.config.get('lw_preserve_variances', True))[0] * 252
                if cfg.get('mu_convention', 'arithmetic') == 'arithmetic':
                    mu_sample = np.expm1(train_data.values).mean(axis=0) * 252
                else:
                    mu_sample = train_data.mean().values * 252

                tickers_list = list(train_data.columns)
                sector_priors_cfg = cfg.get('sector_priors', {})
                sector_map_cfg    = cfg.get('sector_map', {})
                n_train_obs = len(train_data)
                N_assets = len(tickers_list)
                js_c = _js_base_intensity(n_train_obs, N_assets)[0] * max(1.0, np.sqrt(60 / max(n_train_obs, 1)))
                mu_prior = np.array([
                    sector_priors_cfg.get(sector_map_cfg.get(t, ''), 0.07)
                    for t in tickers_list
                ])
                gamma_js = js_c / (js_c + n_train_obs)
                mu_train_js = (1 - gamma_js) * mu_sample + gamma_js * mu_prior

                vol_avg = float(np.sqrt(np.diag(sigma_train)).mean())
                mu_clip_hi = min(rf_wfo + 1.8 * vol_avg, 0.20)
                mu_train = np.clip(mu_train_js, -0.50, mu_clip_hi)

                from scipy.optimize import minimize
                n_assets = len(tickers_list)
                min_w = cfg.get('min_weight', 0.02)
                max_w = cfg.get('max_weight', 0.20)

                sector_constraints_cfg = cfg.get('sector_constraints', {})
                constraints = _slsqp_constraints(tickers_list, sector_map_cfg, sector_constraints_cfg)

                def neg_sharpe(w):
                    w = np.clip(w, 0, None)
                    if w.sum() < 1e-8: return 1e10
                    w /= w.sum()
                    r = np.dot(w, mu_train)
                    v = np.sqrt(np.dot(w, np.dot(sigma_train, w)))
                    return -(r - rf_wfo) / (v + 1e-10)

                bounds = [(min_w, max_w)] * n_assets
                best_w = np.ones(n_assets) / n_assets
                best_v = np.inf
                for s in range(8):
                    w0 = np.random.default_rng(s * 7).dirichlet(np.ones(n_assets))
                    w0 = np.clip(w0, min_w, max_w); w0 /= w0.sum()
                    try:
                        res = minimize(neg_sharpe, w0, method='SLSQP',
                                       bounds=bounds, constraints=constraints,
                                       options={'maxiter': 800, 'ftol': 1e-10})
                        if res.success and res.fun < best_v:
                            best_v = res.fun
                            best_w = np.clip(res.x, 0, None); best_w /= best_w.sum()
                    except Exception:
                        continue

                if not np.isfinite(best_v):
                    print(f"  [WFO WARN] Window {train_start}: no solver converged — equal weights used")
                test_simple = np.expm1(test_data)   # log -> simple returns before aggregating
                oos_ret = (test_simple * best_w).sum(axis=1).copy()
                if prev_weights is not None and len(oos_ret) > 0:
                    turnover = float(np.sum(np.abs(best_w - prev_weights)))
                    oos_ret.iloc[0] = oos_ret.iloc[0] - tx_rate * turnover
                prev_weights = best_w.copy()
                port_returns.append(oos_ret)
                bench_returns.append(test_simple[self.benchmark])
                windows.append(f"{train_start}→{test_end}")
                all_weights.append(best_w)

            except Exception as e:
                print(f"  [WFO WARN] Window {train_start}: {e}")

            start_idx += self.rebal_days

        if not port_returns:
            print("  [WFO WARN] No OOS window completed")
            return {}

        port_series  = pd.concat(port_returns).sort_index()
        bench_series = pd.concat(bench_returns).sort_index()

        def compute_metrics(ret_series, rf=rf_wfo):
            ann_ret  = ret_series.mean() * 252
            ann_vol  = ret_series.std() * np.sqrt(252)
            sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
            downside = np.sqrt(np.mean(np.minimum(ret_series.values, 0.0) ** 2)) * np.sqrt(252)
            sortino  = (ann_ret - rf) / downside if downside > 0 else 0
            cum      = (1 + ret_series).cumprod()
            roll_max = cum.cummax()
            dd       = (cum - roll_max) / roll_max
            max_dd   = dd.min()
            calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
            var_95 = float(np.quantile(ret_series, 0.05))
            cvar_95 = float(ret_series[ret_series <= var_95].mean()) if (ret_series <= var_95).any() else var_95
            return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                        sortino=sortino, max_dd=max_dd, calmar=calmar,
                        var_95=var_95, cvar_95=cvar_95)

        port_metrics  = compute_metrics(port_series)
        bench_metrics = compute_metrics(bench_series)

        print(f"  Completed OOS windows: {len(windows)}")
        print(f"  OOS period: {port_series.index[0].date()} → {port_series.index[-1].date()}")
        print()
        print(f"  {'Metric':25s} {'Portfolio':>15s} {'Benchmark':>15s}")
        print(f"  {'-'*55}")
        for k in ['ann_ret', 'ann_vol', 'sharpe', 'sortino', 'max_dd', 'calmar', 'var_95', 'cvar_95']:
            pm = port_metrics[k]
            bm = bench_metrics[k]
            suffix = '%' if k in ['ann_ret', 'ann_vol', 'max_dd', 'var_95', 'cvar_95'] else ''
            pm_str = f"{pm*100:.2f}%" if suffix else f"{pm:.3f}"
            bm_str = f"{bm*100:.2f}%" if suffix else f"{bm:.3f}"
            print(f"  {k:25s} {pm_str:>15s} {bm_str:>15s}")

        return dict(
            port_metrics=port_metrics,
            bench_metrics=bench_metrics,
            port_returns=port_series,
            bench_returns=bench_series,
            windows=windows,
            protocol=wfo_protocol,
            oos_gap_days=oos_gap_days,
            transaction_cost_bps=tx_bps
        )

# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  USER CONFIG  ██████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {

   # ────────────────────────────────── ───────────────────────────────────
    # ENTER YOUR ETFs HERE (Yahoo Finance tickers or Excel names)
    # NOTE: the FIRST ticker is used as the benchmark for the HMM
    # (ideally a broad equity ETF such as MSCI World).
    # WARNING: verify every ticker on Yahoo Finance before running. Tickers can change or be
    # delisted (check e.g. STN.PA), and a ticker with no data is silently excluded.
    # ─────────────────────────────────────────────────────────────────────
    "etf_tickers": [
        # BASE — MSCI World (benchmark, ~2009)
        "IWDA.AS",      # MSCI World  [ISIN IE00B4L5Y983]

        # FIXED INCOME (~2009-2012)
        "EUHA.DE",      # Fixed Income HY EUR Short-Term  [IE00BD8D5G25]
        "X15E.MI",      # Fixed Income EUR Gov 15-30y     [LU0290357507]
        "X57E.DE",      # Fixed Income EUR Gov 5-7y        [LU0290357176]

        # COMMODITIES  ( ~2010-2017)
        "AUCO.MI",      # Gold Mining                     [IE00B3CNHG25]
        "IUES.L",       #  Energy Sector [IE00B42NKQ00, ~2010]
        "STN.PA",      # SPDR MSCI Europe Energy rebrand 2024
        "IS0D.DE",      # Oil & Gas E&P                   [IE00B6R51Z18]

        # MATERIALS (~2007-2010)
        "IUMS.L",      #  iShares S&P500 Materials [IE00B4MKCJ84, ~2010]
        "XSPR.L",       # Xtrackers MSCI Europe Materials [LU0292100806, ~2007]

        # UTILITY AND INDEX (~2007-2010)
        "IUUS.L",      # iShares S&P500 Utilities [IE00B4KBBD01, ~2010]
        "STUX.MI",      # SPDR MSCI Europe Utilities      [IE00BKWQ0P07, ~2017]

        # TECH & AI (~2002-2018)
        "IUIT.L",       #  iShares S&P500 IT Sector [IE00B3WJKG14, ~2002]
        "XAIX.DE",      # Xtrackers AI & Big Data         [IE00BGV5VN51, ~2018]

        # MONETARY / CASH EQUIVALENT (~2007)
        "PEU.MI",       # Invesco Euro Cash 3M            [IE00B3BPCH51]
        "XEON.DE",      # Xtrackers EUR Overnight Rate    [LU0290358497]

        # FINANCIAL SECTOR (~2007)
        "BNKE.PA",      # Amundi Euro Stoxx Banks         [LU1829219390]
        "XS7R.DE",      # Xtrackers MSCI Europe Financials[LU0292103651, ~2007]

        # GEOGRAPHIC DIVERSIFICATION (~2007-2011)
        "JREM.L",       # JPM Emerging Markets ESG        [IE00BF4G6Z54]
        "CPXJ.L",       # iShares MSCI Pacific ex Japan   [IE00B52MJY50, ~2011]
    ],
    # ─────────────────────────────────────────────────────────────────────
    # SECTOR MAP — used for constraints and reporting
    # Modify based on your actual ETFs
    # ─────────────────────────────────────────────────────────────────────
    "sector_map": {
        "IWDA.AS":  "MSCI World",
        "EUHA.DE":  "Fixed Income",
        "X15E.MI":  "Fixed Income",
        "X57E.DE":  "Fixed Income",
        "AUCO.MI":  "Commodities",
        "IUES.L":   "Commodities",
        "STN.PA":  "Commodities",
        "IS0D.DE":  "Commodities",
        "IUMS.L":  "Materials",
        "XSPR.L":   "Materials",
        "IUUS.L":  "Utility and index",
        "STUX.MI":  "Utility and index",
        "IUIT.L":   "Tech & AI",
        "XAIX.DE":  "Tech & AI",
        "PEU.MI":   "Monetary",
        "XEON.DE":  "Monetary",
        "BNKE.PA":  "Financial Sector",
        "XS7R.DE":  "Financial Sector",
        "JREM.L":   "Emerging Markets",
        "CPXJ.L":   "Asia Pacific",
    },

   # ─────────────────────────────────────────────────────────────────────
    # DATA — SOURCE SELECTION
    # Option A: FactSet Excel file (leave as None for Yahoo Finance)
    # Option B: Automatic Yahoo Finance
    # ─────────────────────────────────────────────────────────────────────
    "excel_file_path":  None,
    "excel_sheet":      "Sheet1",
    "excel_date_col":   "Date",
    "period_years":     5,
    "benchmark_period_years": 20,   # years for HMM benchmark (IWDA.AS since 2009)

    # ─────────────────────────────────────────────────────────────────────
    # HMM MODEL
    # ─────────────────────────────────────────────────────────────────────
    "n_regimes":            3,      # KEEP 3 with historical series <3 years. Never >4 with <300 obs
    "hmm_iterations":       150,    # iterations Baum-Welch
    "volatility_window":    10,     # rolling volatility (in days)
    "auto_select_regimes":  False,  # HOLD False — with short series, BIC selects too many regimes.
    "random_state":         42,
    # ── ENSEMBLE & STABILITY ────────────────────────────────────────────────
    "hmm_n_ensemble":       10,     # number of HMM models (seed 0..9) — weighted average
                                    # higher = more stable, slower (10 recommended)
    "regime_prob_floor":    0.03,   # minimum floor for each regime (3%)
                                    # avoids estimates like 97% sideways over a 5-year horizon
                                    # recommended range: 0.03 (aggressive) – 0.10 (conservative)

   # regime_smoothing_window: number of days to average (5 = 1 week)
    #   1 = original behavior (no smoothing, maximum sensitivity)
    #   5 = recommended (filters daily noise, responsive within ~1 week)
    #  10 = conservative (changes regime only after 2 weeks of signal)
    # regime_smoothing_decay: weight decreasing for older days
        #   0.0 = simple average | 0.3 = slight weighting towards recent days (recommended)
    "regime_smoothing_window": 21,   # smoothing days (1=off, 5=short, 21=monthly RECOMMENDED)
    "regime_smoothing_decay":  0.1,  # mild decay with K=21 (0.0=pure average, 0.1=mild recency)

    # ─────────────────────────────────────────────────────────────────────
    # OPTIMIZATION MVO
    # ─────────────────────────────────────────────────────────────────────
    "risk_free_rate":   0.03,       # annual risk-free rate (update to the current ECB deposit-facility rate)
    "min_weight":       0.02,       # minimum ETF weight — 2% floor avoids unwanted zero weights

   # ─────────────────────────────────────────────────────────────────────
    # PERFORMANCE BOOST — 3 independent levers, usable individually or together
    # ─────────────────────────────────────────────────────────────────────

    # LEVER 1 — MOMENTUM TILT (Jegadeesh & Titman 1993)
    # 0.0 = disabled | 0.10-0.15 = moderate | 0.20-0.30 = strong
    "momentum_tilt":        0.08,
    "momentum_window_long":  252,
    "momentum_window_short":  21,

    # LEVER 2 — RETURN TILT on the MVO objective
    # Shifts the optimization from pure Sharpe toward higher return.
    # 0.0 = pure max Sharpe (default) | 0.20 = recommended | 1.0 = pure max return
    # 0.0  = pure max Sharpe (risk/return efficient but does not maximize absolute return)
    # 0.20 = recommended: 80% Sharpe + 20% pure return. Adds ~+1-3% return.
    # 0.40 = moderately aggressive: more return, slightly lower Sharpe.
    # 1.0  = pure max return (completely ignores risk — not recommended)
    # IMPORTANT: to activate return_tilt, objective_mode MUST be "blended".
    "return_tilt":          0.0,   # v7 FIX: was 13.0 (out of range). Valid range: [0.0, 1.0].
    # Methodological objective selection:
    # "max_sharpe" → pure max Sharpe (ignores return_tilt — use for comparison)
    # "blended"    → ACTIVATES return_tilt (use this for a return boost)
    # "max_return" → forces return_tilt=1.0 (pure max return, not recommended)
    # "vol_target" → maximizes return subject to a vol constraint (uses vol_target)
    "objective_mode":       "max_sharpe",  # v7 FIX: was "max_sharpe" → ignored return_tilt

    # LEVER 3 — VOLATILITY TARGET (alternative to return_tilt)
    # Maximizes return with an explicit volatility constraint.
    # None = disabled | 0.13 = 13% vol target | 0.15 = 15% vol target
    # Use EITHER return_tilt OR vol_target, not both together.
    # EXPECTED EFFECT: maximum return compatible with the vol target
    "vol_target":           None,
    "max_weight":       0.22,       # v5 FIX: was 0.16 (too tight with 20 ETFs + 2% min). Increased to 22%.

    # Sector constraints: {sector_name: (min_weight, max_weight)}
    # FEEL FREE TO MODIFY based on your strategy
    # NOTE: sector names must match EXACTLY
    # the values in the sector_map above (including capitalization and spaces).
    # ─────────────────────────────────────────────────────────────────────
    # SECTOR CONSTRAINTS — per-sector constraints (min%, max%)
    # v7 FIX: restored meaningful constraints (were all (0,1) = no constraint).
    # (0,1) is equivalent to no constraint → the optimizer can concentrate 30%
    # in a single sector (e.g. Commodities) if momentum is favorable.
    # Advantages of constraints: enforce structural diversification, reduce concentration.
    # Drawback: may slightly lower the optimized Sharpe.
    # To test without constraints: set all to (0.00, 1.00) or comment out the dict.
    # NOTE: names must match the values in sector_map EXACTLY.
    # ─────────────────────────────────────────────────────────────────────
    "sector_constraints": {
        "MSCI World":        (0.10, 0.25),  # still global but not dominant
        "Fixed Income":      (0.06, 0.15),  # defense against bear markets
        "Commodities":       (0.06, 0.18),  # cap avoids 30% concentration
        "Utility and index": (0.04, 0.18),  # stable dividends
        "Materials":         (0.00, 0.15),  # cyclical — no minimum
        "Tech & AI":         (0.05, 0.18),  # structural growth
        "Monetary":          (0.02, 0.08),  # minimum liquidity
        "Financial Sector":  (0.04, 0.15),
        "Emerging Markets":  (0.00, 0.10),
        "Asia Pacific":      (0.00, 0.10),
    },

    # ─────────────────────────────────────────────────────────────────────
    # REGIME WEIGHTING — choose the optimization mode
    #
    # NEW JSP MODES (Joint Stochastic Programming):
    #   "jsp"      → maximizes the average Sharpe across 1000 simulated trajectories (RECOMMENDED)
    #   "jsp_var"  → minimizes the average VaR across 1000 trajectories (best for Max Drawdown)
    #   "jsp_cvar" → minimizes the average CVaR (explicit tail risk)
    #
    # CLASSIC MODES (backup):
    #   "blend"          → mix current + stationary
    #   "current"        → instantaneous probabilities (more reactive)
    #   "stationary"     → long-run distribution (more stable)
    #   "custom"         → uses the weights defined in custom_regime_weights
    # ─────────────────────────────────────────────────────────────────────
    "weighting_mode":   "jsp",          # RECOMMENDED: use JSP

    # ── JSP PARAMETERS ───────────────────────────────────────────────────
    "jsp_n_sequences":  1500,   # simulated trajectories (500-2000; 1000 optimal)
    "jsp_seq_length":   756,    # simulation horizon in trading days (756 ≈ 3 years, 1260 ≈ 5 years).
                                # A longer horizon lets the Markov chain converge toward the long-run
                                # regime mix, so today's regime matters less for the final weights.
    "var_confidence":   0.05,   # VaR confidence level for jsp_var (default 5%)
    "cvar_alpha":       0.05,   # CVaR tail level (used in jsp_cvar)

    # ── REGIME STARTING MODE ─────────────────────────────────────
    # Controls how the Markov simulation is initialized.
    #
    # "today_probs"  RECOMMENDED for 4-5 year investors.
    #                Uses TODAY's probability distribution (full vector,
    #                NOT argmax). With seq_length=1260 the simulation starts from the
    #                current regime and gradually converges to the long-run mix.
    #                DYNAMIC (varies with the market) and BALANCED (5 years).
    #                Example 20/03 (Sideways 90%): projects 5 years → ~37% Sideways.
    #                Example 16/03 (Bull 90%):     projects 5 years → ~49% Bull.
    #                The difference is realistic and not excessive.
    #
    # "blended"      Compromise: mix of today and the stationary distribution π.
    #                start = (1-α)*today + α*π  with α=stationary_blend_alpha.
    #                Useful if you want to further reduce sensitivity.
    #
    # "current"      ONLY for tactical paper trading (horizon of weeks).
    #                Uses the argmax of the current regime smoothed over K days.
    #                Very sensitive portfolio: NOT suitable for 4-5 years.
    "regime_start_mode":      "today_probs",  # "today_probs" | "blended" | "current"
    "stationary_blend_alpha": 0.60,           # weight of the stationary distribution in the blend (0.3-0.70)

    # NOTE: the former 'phase1_days' and 'temporal_weight_tau' settings were removed.
    # They were documented as active but had no effect on the results: a single Markov
    # chain is simulated from today's smoothed regime distribution, and μ / Σ are
    # estimated with uniform weights.

    # ── ERC BLEND (structural diversification) ─────────────────────────
    # Applies Equal Risk Contribution as a post-MVO/JSP diversification constraint.
    # 0.0 = pure MVO/JSP | 0.05 = light | 0.20 = strong (not recommended with 15 ETFs)
    #
    # v3 FIX: was 0.20 → secondary cause of the EW bias.
    # With 15 ETFs of similar risk (after LW+MP shrinkage), ERC converges
    # almost to equally-weighted → a 20% blend is equivalent to +20% weight on the EW solution.
    # With JS_C=20 (main fix) the MVO weights already differentiate well.
    # 0.05 keeps a slight diversification benefit without distorting.
    "erc_blend":        0.0,  # v3 FIX: was 0.20

    # ─────────────────────────────────────────────────────────────────────
    # ANTI-OVERFITTING — 3 techniques to reduce IS-OOS bias
    # ─────────────────────────────────────────────────────────────────────

    # TECHNIQUE 1 — Marchenko-Pastur denoising (post Ledoit-Wolf)
    # Ledoit-Wolf target. True (recommended): shrink the CORRELATION matrix and keep every asset's own
    # variance. False: legacy scikit-learn behaviour (shrinks toward mu*I), which inflates the risk of
    # low-volatility assets such as money-market or short-duration bond ETFs.
    "lw_preserve_variances": True,
    # Convention of the expected returns fed to the optimizer.
    # "arithmetic" (recommended): regime means are averages of SIMPLE returns, consistent with the
    #   arithmetic sector priors and with the mean-variance framework.
    # "log": legacy behaviour (means of log returns, about sigma^2/2 per year lower).
    "mu_convention": "arithmetic",
    # Removes noise eigenvalues from the covariance matrix.
    # Applied intra-regime (mitigates the i.i.d. problem) with adaptive blending.
    # True = active | False = use Ledoit-Wolf only
    "use_mp_denoising":     True,

    # TECHNIQUE 2 — Nested CV (internal 70/30 split in the JSP simulation)
    # The 15 multi-start candidates are selected by performance on the
    # validation set (30%), not on the training set — eliminates selection bias.
    # FIXED parameters: nested_cv_ratio=0.70 — do not optimize on the data.
    "use_nested_cv":        True,
    "nested_cv_ratio":      0.70,   # fixed split — do not touch

    # TECHNIQUE 3 — L2 Turnover Penalty in the JSP objective
    # Penalizes the distance from the previous window's weights.
    # lambda_L2 = 0.10 (Jorion 1986 standard) — do not optimize on the data.
    # 0.0 = disabled | 0.05 = light | 0.10 = standard | 0.20 = strong
    #
    # v3 FIX: was 0.50 → tertiary cause of the EW bias.
    # With lambda=0.50 and w_ref=EW (first WFO run), the penalty pulls the weights
    # toward EW with strength 0.50 * ||w - w_EW||^2.
    # The penalty gradient at distance d=0.10 is 0.10 (versus Sharpe ~0.02-0.05).
    # With 0.05 the penalty is 10x weaker than the Sharpe gradient.
    # Correct range: 0.05 (default) - 0.15 (if you want weight stability in the WFO).
    "l2_turnover_penalty":  0.0,   # v3 FIX: was 0.50 (10x excessive)
    # Explicit transaction cost (bps per unit of turnover)
    # Applied in both the JSP and the Walk-Forward OOS.
    "transaction_cost_bps": 5.0,
    # Your CURRENT portfolio as {ticker: weight} (None = start from scratch). When given, the JSP
    # charges transaction_cost_bps and l2_turnover_penalty against the distance from these weights;
    # without it those two settings have no effect on the JSP.
    "current_weights": None,

    # ── HMM VALIDATION ──────────────────────────────────────────────────
    "run_hmm_validation": True,  # Shows Train/Test gap and Silhouette Score

    # ─────────────────────────────────────────────────────────────────────
    # PRIOR FORWARD-LOOKING PER SECTOR (James-Stein shrinkage target)
    # Annualized expected returns by asset class.
    # Source: Damodaran (2024), AQR (2024), Research Affiliates (2024).
    # Change these values if you have different views on the sectors.
    # ─────────────────────────────────────────────────────────────────────
    # NOTE: these values are automatically used as the shrinkage target
    # in get_regime_statistics(). Do not change the sector names —
    # they must match the values in sector_map exactly.
    #
    # v3 FIX: more differentiated priors (range widened from 3%-10% → 2%-13%).
    # With JS_C=20 and priors compressed in [3%-10%], shrinkage pulled all
    # μ toward an almost identical prior → reduced cross-sectional dispersion.
    # A wider range increases differentiation → stronger gradient.
    "sector_priors": {
        # v5 FIX: names aligned to sector_map (same rule as sector_constraints).
        # "Utilities and Index" → "Utility and index" | "Monetary ETF" → "Monetary"
        # "Materials" was already present — only the name was correct in sector_map.
        "MSCI World":        0.10,   # equity broad: ERP 5% + rf 3%
        "Fixed Income":      0.03,   # aligned with EUR IG yields 2026
        "Commodities":       0.065,  # energy commodities cycle
        "Utility and index": 0.085,  # v5 FIX: was "Utilities and Index" → silent mismatch
        "Materials":         0.10,   # pays more than commodities, less than tech
        "Tech & AI":         0.12,   # premium AI capex supercycle
        "Monetary":          0.025,  # v5 FIX: was "Monetary ETF" → silent mismatch
                                     # ECB cutting: money market falls toward 2.5%
        "Financial Sector":  0.09,   # EUR bank spreads still wide
        "Emerging Markets":  0.09,
        "Asia Pacific":      0.07,
    },
    # Optional anti-look-ahead: dated schedule of priors (as-of).
    # The model uses ONLY the record with effective_date <= window date.
    # Example:
    # [{"effective_date": "2024-01-01", "priors": {...}}, ...]
    "prior_schedule": None,
    "strict_prior_schedule": False,

    # ── REGIME MU FLOORS (optional) ──────────────────────────────────────
    "regime_mu_floors": {
        "Sideways/Kangaroo": -0.02,   # floor -2% p.a. (vs historical -4.8%)
        "Bear":              -0.12,   # floor -12% p.a. (vs historical -13.4%)
        "Bull":               0.04,   # floor +4% p.a. — does not affect optimistic estimates
    },

    # ── BLEND PARAMETERS (used only if weighting_mode != "jsp") ─────────
    "blend_alpha":      0.4,        # 40% current + 60% stationary

    # Used only if weighting_mode = "custom"
    # Enter your personal view on the regimes here
    "custom_regime_weights": {
        "Bull":               0.55,
        "Sideways/Kangaroo":  0.30,
        "Bear":               0.15,
    },

    # ─────────────────────────────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────────────────────────────
    "show_plots":       True,
    "save_plot_path":   "portfolio_dashboard.png",
    "output_csv":       "portfolio_allocation.csv",

    # ─────────────────────────────────────────────────────────────────────
    # WALK-FORWARD VALIDATION (optional)
    # Run an OOS backtest to validate the JSP weights on unseen data
    # FIX 10: Added for anti-overfitting diagnostics
    # ─────────────────────────────────────────────────────────────────────
    "run_walkforward":    True,     # True = runs WFO (slower)
    "wfo_train_months":   24,
    "wfo_rebal_months":    3,       # rebalancing frequency (quarterly)
    "wfo_protocol":       "expanding",  # "rolling" | "expanding"
    "oos_gap_days":        5,       # train/test embargo against leakage
    # ── WFO START DATE (optional) ──────────────────────────────────────────
    # Forces the start of the OOS period to a specific date.
    # Useful for: comparing with an external backtest that starts on a precise date.
    # None = automatic behavior (starts train_days after the first obs).
    # Example: "2021-01-01" to capture the full 2021-2026 cycle.
    "wfo_start_date": None,         # e.g. "2021-01-01" or None
}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    optimizer = HMMPortfolioOptimizer(CONFIG)
    allocation = optimizer.run()
