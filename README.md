# Trading-with-Wavelet-Based-Deep-Reinforcement-Learning
Collecting workspace information# Algorithmic Trading Research Platform

A comprehensive research platform for algorithmic trading using reinforcement learning and classical machine learning methods, with advanced wavelet-based signal denoising and feature extraction.

## Overview

This research platform implements and compares multiple algorithmic trading approaches:
- **Reinforcement Learning**: PPO, A2C, DQN with advanced neural architectures
- **Classical ML**: XGBoost, Random Forest, Logistic Regression
- **Baseline Strategies**: Buy & Hold, Moving Average Crossover
- **Signal Processing**: Multi-wavelet denoising with effectiveness analysis

## Key Features

### Advanced Signal Processing
- **Multi-wavelet Families**: Daubechies, Symlets, Coiflets, Biorthogonal
- **Denoising Analysis**: SNR improvement, noise reduction metrics, frequency domain analysis
- **Statistical Validation**: T-tests, p-values, significance testing

### Sophisticated Neural Architecture
- **Advanced Feature Extractor**: Multi-head attention + BiLSTM + positional encoding
- **Architecture Components**:
  - Positional encoding for temporal relationships
  - Multi-head self-attention mechanism
  - Bidirectional LSTM for sequence modeling
  - Layer normalization and dropout regularization

### Comprehensive Evaluation
- **13 Financial Metrics**: Alpha, Beta, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, RoMaD, CAGR, etc.
- **Statistical Analysis**: T-statistics, p-values, significance tests
- **Temporal Data Splitting**: Prevents data leakage with proper train/validation/test splits

### Trading Environment
- **Long-Only Strategy**: Realistic market constraints
- **Transaction Costs**: Spreads and commissions included
- **Position Management**: Proper entry/exit mechanics
- **Performance Tracking**: Real-time balance and trade monitoring

## Configuration

### Hyperparameter Grid
```python
HYPERPARAM_GRID = {
    'PPO': {
        'learning_rate': [3e-4, 1e-4],
        'gamma': [0.95, 0.99],
        'n_steps': [2048],
        'batch_size': [128]
    },
    'A2C': {
        'learning_rate': [1e-4, 5e-5, 3e-4],
        'gamma': [0.95, 0.99],
        'n_steps': [20, 50]
    },
    'DQN': {
        'learning_rate': [1e-4, 5e-5, 3e-4],
        'gamma': [0.95, 0.99],
        'buffer_size': [100000, 50000],
        'exploration_fraction': [0.1, 0.2]
    }
}
```

### Training Configuration
- **Hyperparameter Tuning**: 10,000 timesteps
- **Final Training**: 50,000 timesteps
- **Validation Steps**: 5,000 timesteps

## Data Requirements

Expected CSV format with columns:
- `date`: Trading date (used as index)
- `PRICE`: S&P 500 futures price
- `DIX`: Dark Index indicator
- `GEX`: Gamma Exposure indicator
- `VIX`: Volatility Index

## Architecture Components

### Trading Environment (`SimpleTradingEnv`)
```python
class SimpleTradingEnv(gym.Env):
    # Long-only trading environment with proper reset mechanics
    # Includes spreads, commissions, and position tracking
```

### Advanced Feature Extractor
```python
class AdvancedFeatureExtractor(nn.Module):
    # Multi-head attention + BiLSTM + positional encoding
    # Sophisticated feature extraction for RL agents
```

### Metrics Calculator
```python
def calculate_complete_metrics(balance_history, price_data, benchmark_returns=None):
    # Calculates 13+ financial performance metrics
    # Includes statistical significance testing
```

## Output Structure

### Results Directory
```
results_COMPLETE_WITH_CLASSICAL_METHODS_2/
├── graphs/                 # Performance visualizations
├── csv/                   # Raw results data
├── out_of_sample/         # Test results
├── academic_reports/      # Statistical analysis
├── denoising_analysis/    # Wavelet effectiveness
└── wavelet_analysis/      # Signal processing results
```

### Key Output Files
- `complete_results_with_classical_methods_FIXED.csv`: Main results
- `denoising_effectiveness.csv`: Wavelet analysis
- `statistical_significance.txt`: Significance testing

## Research Methodology

### Temporal Data Splitting
- **Training**: 60% (earliest data)
- **Validation**: 20% (middle period)
- **Testing**: 20% (most recent data)

### Wavelet Processing
```python
def simple_wavelet_denoise(series, wavelet):
    # Level-2 decomposition with soft thresholding
    # Conservative threshold: 0.1 * σ
```

### Feature Combinations
- Single indicators: DIX, GEX, VIX
- Pairs: DIX+GEX, DIX+VIX, GEX+VIX  
- Triple: DIX+GEX+VIX

## Key Innovations

1. **Reproducible Research**: Unique random seeds per experiment
2. **No Data Leakage**: Strict temporal splitting
3. **Realistic Trading**: Transaction costs and market constraints
4. **Multi-Method Comparison**: RL vs Classical vs Baselines
5. **Statistical Rigor**: Significance testing and confidence intervals
6. **Advanced Architecture**: State-of-the-art neural feature extraction

## Dependencies

```python
# Core Libraries
numpy, pandas, matplotlib, scikit-learn
torch, gym, stable-baselines3
pywt, xgboost, seaborn, scipy

# Specific Versions (recommended)
torch>=1.9.0
stable-baselines3>=1.6.0
scikit-learn>=1.0.0
```

## Usage

1. **Prepare Data**: Ensure CSV file matches expected format
2. **Configure Paths**: Update `file_path` and `results_dir`
3. **Run Analysis**: Execute the complete pipeline
4. **Review Results**: Check output directory for comprehensive results

## Performance Metrics

### Trading Metrics
- **Return Metrics**: Total Return, CAGR
- **Risk Metrics**: Volatility, Max Drawdown, Sortino Ratio
- **Risk-Adjusted**: Sharpe Ratio, Alpha, Beta, RoMaD
- **Trading Stats**: Win Rate, Max Loss, Number of Trades

### Statistical Validation
- **Significance**: T-statistics, p-values
- **Comparison**: Paired t-tests between methods
- **Robustness**: Multiple random seeds and cross-validation

This platform provides a comprehensive framework for algorithmic trading research with proper statistical validation and realistic market simulation.
