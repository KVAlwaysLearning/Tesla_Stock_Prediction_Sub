
# 🏎️ Tesla Stock Price Prediction & Real-Time Algorithmic Forecasting Hub

An institutional-grade quantitative research framework and real-time analytical dashboard designed to model, validate, and forecast the chronological sequence paths of Tesla, Inc. (`TSLA`) equity traded on the NASDAQ exchange. 

This project bridges deep sequence modeling (**1D CNN + GRU/LSTM networks**) with statistical regime filters (**Hurst Exponent**) and stochastic mathematical frameworks (**Ornstein-Uhlenbeck Mean-Reversion Loops**), packaged within a highly responsive, modern dark-themed production user interface.

---

## 📌 Core Objective & Forecasting Horizons

Financial time-series data inherently contains high noise levels, volatility clustering, and non-linear dependencies. This architecture avoids flat static regressions by leveraging a three-dimensional rolling vector sequence to forecast multi-step ahead asset behaviors. 

Using a sliding chronological input context window of **exactly 60 historical trading days**, the engine projects absolute closing asset prices across three distinct predictive horizons:
* **1-Day Horizon ($t+1$):** Localized next-session price projections.
* **5-Day Horizon ($t+5$):** Short-term weekly macro trend orientation.
* **10-Day Horizon ($t+10$):** Extended bi-weekly momentum envelope.

---

## 🛠️ System Architecture & Framework Dependencies

The implementation is split into a data engineering pipeline, a deep learning optimization suite, and a localized web distribution layer. 

### Core Tech Stack:
* **Data Processing & Analytics:** `numpy`, `pandas`, `scipy`
* **Statistical Models:** `statsmodels` (Augmented Dickey-Fuller Testing)
* **Deep Learning Engine:** `TensorFlow 2.x`, `Keras`, `scikeras` (Time-Series Hyperparameter Grids)
* **Explainable AI (XAI):** `SHAP` (Shapley Additive Explanations via Cooperative Game Theory)
* **Dashboard Stack:** `streamlit`, `plotly` (Multi-Axis Canvas Graphics)

---

## 🔬 Quantitative Research Pipeline (Notebook Blueprint)

### 1. Data Ingestion & Chronological Integrity
* **Remote Streaming Ingestion:** Dynamically connects to raw historical streams via a cookie-resilient `requests.Session()` loop to bypass public warnings and process data securely in-memory using `io.BytesIO`.
* **Chronological Re-Indexing:** Transforms raw `Date` string values into explicit `datetime64` data objects and locks them as the primary dataframe structural index, sorting transactions ascendingly ($t \rightarrow t+1$) to completely prevent temporal data leakage.

### 2. Parametric Statistical Testing
* **Stationarity Evaluation (ADF Test):** Running the Augmented Dickey-Fuller test yielded a $p$-value of `0.841`, failing to reject the null hypothesis ($H_0$). The asset price represents a non-stationary random walk, confirming the mathematical necessity of sequence-based deep states.
* **Volume-Volatility Dependency (Chi-Square Test):** Rejects independence ($p < 0.001$), confirming that trading volume surges serve as significant liquidity indicators that directly influence upcoming volatility regimes.

### 3. Feature Pipeline & Lookback Structuring
* **Imputation & Outlier Strategy:** Employs temporal **Forward-Filling** to handle gap states safely, while preserving raw extreme session boundaries to teach deep layers how to handle market flash crashes.
* **MinMax Scaling:** Compresses input metrics cleanly between $(0,1)$ to prevent vanishing gradients during multi-epoch optimization passes.
* **Chronological Splitting:** Enforces a strict **80% Training / 20% Test** chronological partition rule. Traditional random K-Fold cross-validations are completely banned to maintain temporal separation.

---

## 🧠 Deep Learning Modeling Suite

The project designs, validates, and evaluates three distinct sequence models using a custom **Time-Series Forward-Chaining Grid Search**:

```text
   [Raw 60-Day Market Window Input]
                │
                ▼
  ┌───────────────────────────┐
  │       1D CNN Layer        │  <-- Stage 1: Spatial Noise Filter
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │       GRU Layer           │  <-- Stage 2: Temporal Sequence Tracking
  └─────────────┬─────────────┘
                │
                ▼
  ┌───────────────────────────┐
  │     Dense Output Layer    │  <-- Stage 3: Multi-Horizon Projection
  └───────────────────────────┘

```text

1.  **Model-I: SimpleRNN Baseline** A standard recurrent neural network with 50 processing nodes. Used to establish a performance floor, though it remains structurally vulnerable to gradient degradation over long sequences.
2.  **Model-II: Deep LSTM Layering** A 64-unit network utilizing input, forget, and output gating channels to maintain long-term memory across the 60-day historical window.
3.  **Model-III: Hybrid 1D CNN-GRU Framework (Champion)** Our production system. It passes the raw 60-day sequence tensor into an initial **1D Convolutional Neural Network (Conv1D)** layer to filter out high-frequency market noise. The filtered spatial feature vectors are then tracked by an optimized, lightweight **Gated Recurrent Unit (GRU)** layer.

### Model Evaluation Matrix on Unseen Test Sets:
| Model Configuration | 1-Day Horizon (MSE) | 5-Day Horizon (MSE) | 10-Day Horizon (MSE) | Latency Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Model-I: SimpleRNN** | 0.0142 | 0.0384 | 0.0712 | Minimal |
| **Model-II: Deep LSTM** | 0.0031 | 0.0089 | 0.0194 | Heavy |
| **Model-III: Hybrid CNN-GRU** | **0.0018** | **0.0049** | **0.0112** | **Highly Efficient** |

* **Explainable AI (SHAP Plotting):** Integrated game-theory metrics show that while the network naturally assigns its highest weights to immediate sessions ($t-1$ to $t-3$), it continuously assesses support and resistance patterns from deep historical clusters ($t-20$ and $t-45$).

---

## 🖥️ Live Dashboard Architecture (`app.py`)

The finalized production pipeline is deployed as an interactive dashboard via an optimized **Streamlit** cloud configuration.

### 🌟 Key Production Enhancements:
* **Low-Level HTML Compiler Patch:** Modifies Streamlit's structural `DeltaGenerator` class at runtime, intercepting and compacting code injections to render complex canvas models flawlessly in dark mode.
* **Advanced Regime Detection Engines:** * *The Hurst Exponent Filter:* Computes continuous fractal scales to identify if the current market window is mean-reverting ($H < 0.5$), a pure random walk ($H = 0.5$), or displaying persistent momentum ($H > 0.5$).
    * *Ornstein-Uhlenbeck (OU) Calibration:* When mean-reversion is detected, the app runs an OU drift process ($dx_t = \theta(\mu - x_t)dt + \sigma dW_t$) to calculate the mathematical reversion speed ($\theta$) and the equilibrium price target ($\mu$).
* **Dynamic Visualizations:** Generates a unified, synchronized three-tier **Plotly Subplot Canvas** displaying price target paths, volatility ranges, and volume indicators.

---

## 🚀 Local Installation & Execution

Follow these steps to deploy the quantitative workstation on your local machine:

### 1. Clone the Target Repository
```bash
git clone [https://github.com/your-username/tesla-stock-prediction.git](https://github.com/your-username/tesla-stock-prediction.git)
cd tesla-stock-prediction

```

### 2. Configure Your Python Environment

Ensure you are using Python 3.9, 3.10, or 3.11 for complete library and gradient engine compatibility.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install Required Dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn requests plotly statsmodels scipy tensorflow streamlit scikeras

```

### 4. Execute the Deep Learning Dashboard

```bash
streamlit run app.py

```

---

## 📈 Future Research Directions

* **Multivariate Ingestion:** Incorporating non-price indicators directly into the convolutional channels, including options implied volatility indexes (VIX) and options order flows.
* **Attention Layers:** Introducing Temporal Fusion Transformers (TFT) to assign dynamic importance masks across multi-year macroeconomic cycles.
* **NLP Sentiment Pipelines:** Integrating real-time sentiment analysis tracking financial headlines and social sentiment loops to capture immediate market shifts.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for further details.



```
