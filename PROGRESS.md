# Project progress

Shared log of what has been decided, what has been done and what came out of it.
Newest entries go at the top of each section. Mark anything not yet agreed by the group as **(proposed)**.

## Key dates

| Date | What |
|---|---|
| 7 Nov | Group registration closes |
| 29 Nov | Project report deadline (peer-graded, 30%) |
| 4 Dec, 7–11 Dec | Presentation + oral exam (70%), 15 min + questions |

## Research question

How fast do different car models lose value, and how much of the price is explained by brand vs. model?

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-04 | Separate held-out test set of 10,000 rows, disjoint from the training set | Gives an honest out-of-sample accuracy measure (error in £) in addition to LOO. Used only once, at the end. |
| 2026-11-04 | Cleaning rules below agreed | See the table under Data. |
| 2026-11-04 | Simple random sample of 10,000 rows with a fixed seed | Everyone works on the same data, and the real brand/model mix is kept. |
| 2026-11-04 | Mileage enters linearly, in units of 10,000 miles | Fits better than log(mileage): see "Mileage: linear vs log" below. |
| 2026-11-04 | Model `log(price)`, not `price` | Prices are skewed and depreciation is roughly a percentage per year, so effects are multiplicative. A coefficient β means a factor e^β change in price. |
| 2026-11-04 | Main priors are weakly informative. Vague and informative priors are used only for the sensitivity analysis. | Weakly informative priors rule out absurd prices without favouring any particular answer. The other two sets show whether conclusions depend on the prior. |
| 2026-11-04 | Build models step by step: M1 pooled → M2 brand → M3 brand/model → M4 varying age slopes | Each step adds one idea that can be tested. M1 vs M3 gives the required non-hierarchical vs hierarchical comparison. |
| 2026-11-04 | Every model goes through the full loop (see checklist below) before the next one is added | The course requires diagnostics, posterior predictive checks and sensitivity analysis for all models. |
| 2026-11-04 | Python + Bambi (on PyMC) + ArviZ | The existing code is Python, and Bambi is one of the recommended tools. |

## Data

Source: Kaggle [`adityadesai13/used-car-dataset-ford-and-mercedes`](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes),
UK listings scraped in 2020. `src/data.py` loads the nine per-brand files: 97,711 rows after removing exact duplicates and the impossible year 2060.

Structure: listing → model (195) → brand (9). Each model name belongs to exactly one brand, so the hierarchy is cleanly nested.
On log-price, brand alone explains ~32% of the variance and brand + model ~61%.

### Cleaning rules (agreed)

| Rule | Rows affected | Reasoning |
|---|---|---|
| Keep age ≤ 15 years | drops 168 (0.2%) | A few 1970–2004 cars, where classic-car pricing works differently. |
| Drop rows with missing `engineSize` | drops 266 (0.3%) | `engineSize` is a predictor in M1. |
| Drop `fuelType` Other / Electric | drops 252 (0.3%) | Too few cars to estimate a separate effect. |
| Drop `transmission` Other | drops 9 | Same as above. |
| Keep mpg > 100 | 568 rows, all kept | 507 of them are hybrids (plug-in mpg figures are genuinely high). mpg is not used as a predictor, so they do no harm. |
| Keep mileage < 100 | 2,118 rows, all kept | Almost all are 0–1 years old: pre-registered or nearly new cars, so real listings. They are a problem only for `log(mileage)`, so mileage is used linearly instead (next row). |
| Use mileage in units of 10,000 miles, linear | — | Avoids log(0) and the very high leverage of near-zero mileage. Interpretation: each 10k miles changes the price by a factor e^β. Check the fit in the posterior predictive check. |
| Keep price extremes (£450–£160k) | — | Real cars. If they cause misfit, try Student-t residuals instead of dropping them. |
| Don't use `mpg` or `tax` as predictors | — | Both depend mostly on engine size and fuel type, which are already in the model. |
| Keep all models, including small ones | — | Partial pooling is designed for small groups. Their shrinkage towards the brand mean is a result worth showing. |

### Mileage: linear vs log

Least-squares check (not Bayesian) on the cleaned full data: `log(price) ~ age + engineSize + fuel + transmission + f(mileage)`.
The table shows the mean residual by mileage band, where 0 means no systematic error.

| Mileage band | Rows | Linear (10k mi) | log(mileage+1) | log(mileage+1000) |
|---|---|---|---|---|
| 0–100 | 2,401 | +0.01 | −0.13 | −0.10 |
| 100–1k | 2,408 | +0.06 | +0.01 | −0.03 |
| 1k–60k | ~86k | within ±0.03 | within ±0.04 | within ±0.02 |
| 60k–100k | 5,215 | +0.02 | −0.09 | −0.08 |
| 100k+ | 709 | −0.04 | −0.31 | −0.30 |
| Residual sd | | 0.221 | 0.227 | 0.224 |

- Linear means "each 10k miles costs the same percentage". The fit gives about 4.7% per 10k miles.
- The log versions overpredict high-mileage cars by ~30% and underpredict new ones.
- Age and mileage are correlated (r = 0.75), so their coefficients mean "holding the other fixed" and their posteriors will be correlated. Centring helps the sampler.

### Subsample (agreed, done)

- A simple random sample of **10,000 rows** from the cleaned data, with a fixed seed so everyone gets the same rows.
- A random sample keeps the real brand and model proportions, so M1 isn't distorted. Rare models get only a few rows, which shows partial pooling well.
- If M4 samples too slowly, drop to 5,000.
- A second, disjoint 10k sample is the **held-out test set**. Don't look at it or fit to it until the final model comparison.

**Done (2026-11-04):** `from src.data import train_test; train, test = train_test()` (seed 2026).
`prepare()` in `src/data.py` applies the cleaning rules and adds `log_price` and `mileage_10k`.

- 97,711 → 97,020 rows after cleaning (691 dropped).
- Train: 10,000 rows, 159 car models. 27 of them have fewer than 5 rows, which is good for showing partial pooling.
- Test: 10,000 rows, 164 car models. **9 of them don't appear in train**, so the hierarchical models must predict a new model from its brand level (in Bambi: `sample_new_groups=True`).
- Train summary: log_price mean 9.59 (sd 0.53), age 0–15, mileage 0–212k.
- Age and mileage are **not** centred in the data files. Centre them in the model notebook, using the training-set means for both sets.

## Models and priors

Predictors (M1): age, mileage (10k miles), engineSize, fuelType, transmission. Reference levels: Petrol, Manual.
Age and mileage are centred so that the intercept is the log-price of a typical car.

| Parameter | Vague | **Weakly informative (main)** | Informative |
|---|---|---|---|
| α (intercept, typical car) | Normal(9.5, 10) | **Normal(9.5, 1)** | Normal(9.5, 0.5) |
| β_age (per year) | Normal(0, 10) | **Normal(0, 0.5)** | Normal(−0.15, 0.05) |
| β_mil (per 10k miles) | Normal(0, 10) | **Normal(0, 0.5)** | Normal(−0.05, 0.05) |
| β_eng (per litre) | Normal(0, 10) | **Normal(0, 0.5)** | Normal(0.2, 0.2) |
| Fuel / transmission levels | Normal(0, 10) | **Normal(0, 0.5)** | Normal(0, 0.25) |
| σ (residual sd) | HalfNormal(10) | **Exponential(1)** | HalfNormal(0.5) |
| Group sds (M2–M4, intercepts) | HalfNormal(5) | **HalfNormal(0.5)** | HalfNormal(0.25) |
| Age-slope sd (M4) | HalfNormal(1) | **HalfNormal(0.1)** | HalfNormal(0.05) |

The informative centres are domain guesses ("cars lose ~15% a year", "bigger engines cost a bit more"), not estimates from this data.
The β_mil informative prior (−0.05 per 10k miles) happens to match the least-squares estimate (−0.048). Justify it from an outside source, or change it, so that it isn't using the data twice.

| Model | Formula (Bambi) | Status |
|---|---|---|
| M1 pooled | `log_price ~ age + mileage + engineSize + fuelType + transmission` | not started |
| M2 brand | `... + (1 \| brand)` | not started |
| M3 brand/model | `... + (1 \| brand) + (1 \| brand:model)` | not started |
| M4 varying slopes | `... + (1 + age \| brand) + (1 + age \| brand:model)` | not started |
| M5 (optional) | M4 with Student-t residuals | not started |

### Checklist for every model

- [ ] Priors written down (including any Bambi defaults left unchanged)
- [ ] Prior predictive check: simulated prices plausible?
- [ ] MCMC: settings recorded (chains, draws, tune, target_accept, seed)
- [ ] Convergence: R-hat, ESS, divergences
- [ ] Posterior predictive check: overall, per brand, tails
- [ ] Sensitivity: refit with vague + informative priors (or `priorsense`), and check whether key results change
- [ ] Results noted below

## Results

_Nothing yet._

## Open questions

- Full three-prior refit for every model, or `priorsense` power-scaling for all and full refits only for the final model? Ask in a TA session.

## AI use log

Required for the report (Aalto AI guidelines, points 5 and 6).

| Date | Tool | What it was used for |
|---|---|---|
| 2026-10-20 | Claude Code | Data loader (`src/data.py`) and first EDA notebook |
| 2026-11-04 | Claude Code | `prepare()` / `train_test()` in `src/data.py`, mileage transform check, data-structure summary, discussing prior options, cleaning proposal, this progress file |
