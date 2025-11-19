# Save Jessica - Morty Express Challenge

This project helps you interact with the [Sphinx Morty Express Challenge](https://challenge.sphinxhq.com/).

## Challenge Overview

Your goal is to send 1000 Morties from the Citadel to Planet Jessica through one of three intermediate planets:

- **Planet A (index=0)**: "On a Cob" Planet
- **Planet B (index=1)**: Cronenberg World
- **Planet C (index=2)**: The Purge Planet

The risk for each planet changes dynamically based on the number of trips taken. Your objective is to maximize the number of Morties who arrive safely!

### The Twist

The survival probability for each planet **changes over time** based on the number of trips taken. This is the key to the challenge!

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your API Token

1. Visit <https://challenge.sphinxhq.com/>
2. Request a token with your name and email
3. Create .env file

```bash
echo "SPHINX_API_TOKEN=your_token_here" > .env
```

### 3. Run the Example

```bash
python example.py
```

### Console Output

You'll see survival rates for each planet:

```text
"On a Cob" Planet:
  Survival Rate: 67.50%
  
Cronenberg World:
  Survival Rate: 68.20%
  
The Purge Planet:
  Survival Rate: 65.80%
```

### Visualizations

Several plots will appear showing:

1. **Survival rates over time** - How each planet performs as trips increase
2. **Overall comparison** - Bar chart comparing planets
3. **Moving average** - Trends in survival rates
4. **Risk evolution** - How risk changes (early vs late trips)
5. **Episode summary** - Complete dashboard

## Project Structure

- `api_client.py` - API client with all endpoint functions
- `data_collector.py` - Functions to collect and analyze data
- `visualizations.py` - Functions to visualize challenge data
- `example.py` - Example usage script
- `strategy.py` - Template for building your own strategy

## API Functions

The `SphinxAPIClient` class provides:

- `start_episode()` - Initialize a new escape attempt
- `send_morties(planet, morty_count)` - Send Morties through a portal
- `get_status()` - Check current progress

## Next Steps

### Option 1: Implement a Strategy

Edit `strategy.py` to create your own strategy:

```python
from strategy import run_strategy, SimpleGreedyStrategy

# Run a pre-built strategy
run_strategy(SimpleGreedyStrategy, explore_trips=30)

# Or create your own by subclassing MortyRescueStrategy
```

### Option 2: Custom Script

Create your own Python script:

```python
from api_client import SphinxAPIClient
from data_collector import DataCollector

# Initialize
client = SphinxAPIClient()
collector = DataCollector(client)

# Start episode
client.start_episode()

# Your strategy here...
```

## Checking Your Progress

At any time, check your status:

```python
status = client.get_status()
print(f"Saved: {status['morties_on_planet_jessica']}")
print(f"Lost: {status['morties_lost']}")
print(f"Remaining: {status['morties_in_citadel']}")
```

## API Not Working?

```python
# Test your connection
from api_client import SphinxAPIClient

client = SphinxAPIClient()
status = client.get_status()
print(status)
```

# RLS Sinusoid Strategy

The policy used to decide:

* **which planet** to send Morties to, and
* **how many Morties** to send on each trip,

using a **sinusoidal model + Recursive Least Squares (RLS)** that adapts as we get feedback from the Morty Express API.

In one experiment with:

```python
df = collector.explore_phase_online_adaptive_policy(
    early_explore_p0=0,
    early_explore_p1=0,
    early_explore_p2=0,
)
```

this strategy achieved **~84.4% survival rate** over the episode (exact results will vary per run).

We execute using example.py: 

```python
python example.py
```
---

## High-Level Idea

We assume that for each planet, the **true survival probability** of a Morty is:

* **periodic in time (trip index)**
* with a **known period** (frequency), but **unknown offset, amplitude, and phase**

# Planet Survival Model

For each planet `p`, we model survival as:

```
y(t) ≈ c_p + C_p cos(ω_p t) + S_p sin(ω_p t)
```

## Parameters

- `t` = global trip index (1, 2, 3, …)
- `ω_p = 2π / period_p`
- `c_p` = baseline survival (offset)
- `C_p, S_p` = cosine/sine coefficients encoding amplitude + phase

## Fixed Periods by Planet

Based on prior analysis, we use the following periods:

- **Planet 0** → period 10
- **Planet 1** → period 20
- **Planet 2** → period 200

## Learning Approach

We learn `(c_p, C_p, S_p)` online using Recursive Least Squares (RLS), updating the parameters after each trip based on actual outcomes (survived/lost).

We **fix the period** per planet (based on prior analysis):

* Planet 0 → period 10
* Planet 1 → period 20
* Planet 2 → period 200

Then we **learn** `(c_p, C_p, S_p)` **online with RLS**, updating them after each trip based on what actually happened (survived / lost).

At each step:

1. Use the current sinusoid parameters to **predict survival** on each planet.
2. Pick a planet (explore or exploit).
3. Send some Morties.
4. Observe survival.
5. Update the model **only for that planet** using RLS.

---

## 1. The OnlineSinusoidModel

### 1.1. Parametrization

# Model Assumptions

The model assumes:

```
y(t) ≈ c + C cos(ω t) + S sin(ω t)
```

with a **fixed angular frequency**:

```python
self.period = period
self.omega = 2 * np.pi / period
```

We initialize from interpretable parameters:

* `init_offset` ≈ baseline survival probability
* `init_amplitude` ≈ size of oscillation around that baseline
* `init_phase` ≈ starting phase

They are converted to ((c, C, S)):

```python
c = init_offset
A = init_amplitude
phi = init_phase

C = A * np.cos(phi)
S = -A * np.sin(phi)
```

So:

* `offset` = `c`
* `amplitude` = `√(C² + S²)`
* `phase` is recovered as `atan2(-S, C)`

### 1.2. Prediction

Given a trip index `t`:

```python
phi_t = [1, cos(ωt), sin(ωt)]
y_hat = phi_t @ theta
```

We interpret `y_hat` as a **predicted survival probability**, and clamp it to ([0, 1]):

```python
return np.clip(y_hat, 0.0, 1.0)
```

So `predict(t)` returns:

> “Given what we know so far, what’s the estimated survival probability if we go to this planet on trip t?”

### 1.3. Online RLS Update

Each trip gives us:

* `survived` (0 or 1 if you send 1 Morty; more generally survived morties),
* `morties_sent`,
* the time `t`.

We form an **empirical survival rate**:

```python
y = survived / morties_sent
```

Then apply a **Recursive Least Squares** update:

```python
P = self.P      # 3x3 covariance
θ = self.theta  # [c, C, S]
λ = self.forgetting

phi_t = self._phi(t)
y_hat = phi_t @ θ
e = y - y_hat  # prediction error

denom = λ + phi_t @ P @ phi_t
K = (P @ phi_t) / denom  # gain (3,)

θ_new = θ + K * e
P_new = (1.0 / λ) * (P - np.outer(K, phi_t) @ P)
```
---

## 2. The Policy: `explore_phase_online_adaptive_policy`

This is the high-level strategy that uses one `OnlineSinusoidModel` per planet and the Morty API.

### 2.1. Per-planet models

We create one sinusoid model per planet with fixed period and shared initial guesses:

```python
models = {
    0: OnlineSinusoidModel(period=10.0,  init_offset=0.4, init_amplitude=0.2575, init_phase=0, init_cov=2.0, forgetting=0.999),
    1: OnlineSinusoidModel(period=20.0,  init_offset=0.4, init_amplitude=0.256,  init_phase=0, init_cov=2.0, forgetting=0.999),
    2: OnlineSinusoidModel(period=200.0, init_offset=0.4, init_amplitude=0.256,  init_phase=0, init_cov=2.0, forgetting=0.999),
}
```

### 2.3. Dynamic Morty allocation


1. **Predict** survival for each planet at time `t`:

   ```python
   planet_preds = {p: models[p].predict(t) for p in [0, 1, 2]}
   ```

2. Compute per-planet parameters snapshot for logging:

   ```python
   planet_params_snapshot = {p: models[p].get_params() for p in [0, 1, 2]}
   ```

3. Compute **decision diagnostics**:

   ```python
   sorted_probs = sorted(planet_preds.values(), reverse=True)
   max_prob = sorted_probs[0]
   second_prob = sorted_probs[1]
   gap = max_prob - second_prob
   ```

4. **Choose planet greedily**:

   ```python
   planet = max(planet_preds, key=planet_preds.get)
   planet_name = self.client.get_planet_name(planet)
   pred_prob = planet_preds[planet]
   ```

5. **Decide Morty count** (risk scaling) based on:

   * how early we are in the episode,
   * the **gap** between best and second-best planet,
   * the **absolute confidence** of the best prediction.

   ```python
   if t < 150:
       morties_dyn = 1
   elif gap < 0.05:
       morties_dyn = 1
   elif gap < 0.15:
       morties_dyn = 2
   else:
       morties_dyn = 3

   # Additional safety based on max_prob:
   if max_prob < 0.6:
       morties_dyn = 1
   elif max_prob < 0.7:
       morties_dyn = min(morties_dyn, 2)
   ```

   Intuition:

   * Early, we stay conservative (1 Morty).
   * When one planet looks clearly better → we allow sending 2–3 Morties.
   * If even the best planet’s predicted survival is low → we don’t go all-in.

7. **Update the model only for the chosen planet** with the new observation:

   ```python
   models[planet].update(
       t=t,
       survived=result["survived"],
       morties_sent=result["morties_sent"],
   )

## 3. Results

this policy achieved an **overall survival rate around 84.4%**, by:

* assuming **periodic survival dynamics** per planet with known periods,
* initializing with a **global sinusoid fit**,
* updating offset/amplitude/phase **online via RLS** after each trip,
* picking planets **greedily** based on predicted survival,
* and adjusting Morty count based on **confidence and relative advantage**.

---

## 4. Key Assumptions & Limitations

* Survival probability is **well-approximated by a single sinusoid** per planet (offset + amplitude + phase).
* Periods (10, 20, 200) are **known and fixed**.
* RLS treats survival as a **noisy real value**.

Even with these simplifications, this strategy gives a **simple, interpretable, and reasonably strong** policy.