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
| 2026-11-16 | Group-sd prior for M3 onwards tightened from HalfNormal(0.5) to **HalfNormal(0.3)** | With two group levels, HalfNormal(0.5) put 5.1% of prior predictive prices outside £500–£200k; HalfNormal(0.3) gives 3.7%, about the same as M2 (3.9%). M2 keeps 0.5 as fitted; its results did not depend on it. |
| 2026-11-16 | M3 formula is `(1 \| brand) + (1 \| model)`, not `(1 \| brand:model)` | Bambi builds `brand:model` as the full 9 × 159 cross product: 1,431 intercepts, 1,272 of them for combinations that don't exist. Model names are unique within a brand, so `(1 \| model)` is the same nested model with 159 intercepts. |
| 2026-11-16 | Hierarchical models are sampled with nutpie, centred group effects (`noncentered=False`) and a low-rank mass matrix | Bambi's defaults did not converge for M3 (intercept ESS 11–136, R-hat up to 1.30). The new settings give ESS > 3,400, R-hat 1.00, 0 divergences in ~40s. Needs `pip install nutpie`. See the table in `04_m3_brand_model.ipynb`. |
| 2026-11-04 | **(proposed)** Revise the weakly informative priors for M1: β_age, β_mil ~ Normal(0, 0.1), β_eng ~ Normal(0, 0.3), σ ~ HalfNormal(0.5) | The original set failed the prior predictive check (16% of simulated prices outside £500–£200k). See Results. |
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

Predictors (M1): age, mileage (10k miles), engineSize, fuelType, transmission. Reference levels: Diesel, Automatic (Bambi takes the first level alphabetically).
Age, mileage and engine size are centred so that the intercept is the log-price of a typical diesel automatic car.

| Parameter | Vague | **Weakly informative (main)** | Informative |
|---|---|---|---|
| α (intercept, typical car) | Normal(9.5, 10) | **Normal(9.5, 1)** | Normal(9.5, 0.5) |
| β_age (per year) | Normal(0, 10) | **Normal(0, 0.1)** ~~0.5~~ | Normal(−0.15, 0.05) |
| β_mil (per 10k miles) | Normal(0, 10) | **Normal(0, 0.1)** ~~0.5~~ | Normal(−0.05, 0.05) |
| β_eng (per litre) | Normal(0, 10) | **Normal(0, 0.3)** ~~0.5~~ | Normal(0.2, 0.2) |
| Fuel / transmission levels | Normal(0, 10) | **Normal(0, 0.5)** | Normal(0, 0.25) |
| σ (residual sd) | HalfNormal(10) | **HalfNormal(0.5)** ~~Exponential(1)~~ | HalfNormal(0.25) ~~0.5~~ |
| Group sds (M2–M4, intercepts) | HalfNormal(5) | **HalfNormal(0.3)** from M3 on ~~0.5~~ (M2 was fitted with 0.5) | HalfNormal(0.25) |
| Age-slope sd (M4) | HalfNormal(1) | **HalfNormal(0.1)** | HalfNormal(0.05) |

Struck-through values are the original, agreed priors, replaced after the prior predictive check (proposed). The group-level sds for M2–M4 haven't been checked yet: they get the same treatment when those models are built.

The informative centres are domain guesses ("cars lose ~15% a year", "bigger engines cost a bit more"), not estimates from this data.
The β_mil informative prior (−0.05 per 10k miles) happens to match the least-squares estimate (−0.048). Justify it from an outside source, or change it, so that it isn't using the data twice.

| Model | Formula (Bambi) | Status |
|---|---|---|
| M1 pooled | `log_price ~ age_c + mileage_10k_c + engineSize_c + fuelType + transmission` | full loop complete: priors, sampling, convergence, PPC, sensitivity (`notebooks/02_m1_pooled.ipynb`) |
| M2 brand | `... + (1 \| brand)` | full loop complete: priors, sampling, convergence, PPC, sensitivity (`notebooks/03_m2_brand.ipynb`) |
| M3 brand/model | `... + (1 \| brand) + (1 \| model)` (same nested model as `brand:model`, see Decisions) | full loop complete: priors, sampling, convergence, PPC, sensitivity (`notebooks/04_m3_brand_model.ipynb`, same layout as the M2 notebook). LOO comparison left to the final model comparison. |
| M4 varying slopes | `... + (1 + age \| brand) + (1 + age \| model)` | not started |
| M5 (optional) | M4 with Student-t residuals | not started |

### Checklist for every model

- [x] Priors written down (including any Bambi defaults left unchanged)
- [x] Prior predictive check: simulated prices plausible?
- [x] MCMC: settings recorded (chains, draws, tune, target_accept, seed)
- [x] Convergence: R-hat, ESS, divergences
- [x] Posterior predictive check: overall, per brand, tails
- [x] Sensitivity: refit with vague + informative priors (or `priorsense`), and check whether key results change
- [x] Results noted below


### M1: full Bayesian workflow results (2026-11-12, `notebooks/02_m1_pooled.ipynb`)

- **Sampling:** 4 chains, 1,000 draws, 1,000 tune, `target_accept=0.9`, seed 2026. Runtime 85s (~40 draws/s with Numba).
- **Convergence:** $\widehat{R} = 1.00$ for all parameters, Bulk ESS > 2,000 (min 2,015 on Intercept), Tail ESS > 2,200. Zero divergent transitions. Rank plots show uniform mixing across all 4 chains.
- **Key posterior parameter estimates:**
  - $\alpha = 9.725 \pm 0.006$ (median price of a typical diesel automatic car $\approx £16,730$).
  - $\beta_{\text{age}} = -0.124 \pm 0.002$ (-11.7% depreciation per year).
  - $\beta_{\text{mil}} = -0.045 \pm 0.002$ (-4.4% depreciation per 10,000 miles, closely matching OLS check of -4.7%).
  - $\beta_{\text{eng}} = +0.466 \pm 0.005$ (+59.4% per litre engine size).
  - Manual discount: $\beta_{\text{Manual}} = -0.216 \pm 0.007$ (-19.4% vs Automatic).
  - Residual sd $\sigma = 0.224 \pm 0.002$.
- **Posterior predictive checks:**
  - Overall distribution matches the observed log-price mode well.
  - Tail check: underpredicts slightly in the extreme high end (> £50k: 1.4% observed vs 0.7% simulated).
  - **Per-brand check (crucial):** M1 has massive systematic bias across brands: underpredicts Mercedes (+20.3%), BMW (+16.8%), Audi (+13.5%), and overpredicts Vauxhall (-20.8%) and Hyundai (-16.4%). This strongly motivates **M2 (brand-level varying intercepts)**.
- **Prior sensitivity:**
  - Refit with Vague and Informative priors from the table.
  - Posterior means across all three prior specifications are identical to 3 decimal places (e.g. $\beta_{\text{age}} = -0.124$, $\beta_{\text{mil}} = -0.045$).
  - With $N = 10,000$, the data dominates the likelihood; the weakly informative prior successfully regularizes prior predictive draws without biasing posterior estimates.

### M2: full Bayesian workflow results (2026-11-14, `notebooks/03_m2_brand.ipynb`)

- **Sampling:** 4 chains, 1,000 draws, 1,000 tune, `target_accept=0.95`, `omit_offsets=False`, seed 2026. Runtime ~470s with Numba compilation.
- **Convergence:** $\widehat{R} \le 1.01$ across all parameters, Bulk ESS > 700 (min 736 on Intercept), Tail ESS > 1,200. Zero divergent transitions. Rank plots show uniform mixing across chains.
- **Key posterior parameter estimates:**
  - Grand Intercept: $\alpha = 9.642 \pm 0.069$ (median price of a typical diesel automatic car across brands $\approx £15,400$).
  - $\beta_{\text{age}} = -0.121 \pm 0.001$ (-11.4% depreciation per year).
  - $\beta_{\text{mil}} = -0.048 \pm 0.001$ (-4.7% depreciation per 10,000 miles).
  - $\beta_{\text{eng}} = +0.413 \pm 0.004$ (+51.1% per litre engine size).
  - Manual discount: $\beta_{\text{Manual}} = -0.141 \pm 0.006$ (-13.1% vs Automatic).
  - Hybrid premium: $\beta_{\text{Hybrid}} = +0.216 \pm 0.013$ (+24.1% vs Diesel).
  - Brand SD: $\sigma_{\text{brand}} = 0.194 \pm 0.058$.
  - Residual SD: $\sigma = 0.185 \pm 0.001$.
  - **Residual variance reduction:** Comparing $\sigma$ in M2 (0.185) vs M1 (0.224) reveals a **31.9% reduction in unexplained variance**, perfectly matching the ~32% $R^2$ explained by brand in EDA.
- **Brand intercepts & premiums ($\alpha_{\text{brand}}$):**
  - Audi: $+25.9\%$ ($[+8.8\%, +35.4\%]$ 94% HDI)
  - Mercedes: $+21.2\%$ ($[+5.4\%, +32.2\%]$)
  - BMW: $+10.8\%$ ($[-4.0\%, +22.7\%]$)
  - Volkswagen: $+9.4\%$ ($[-4.6\%, +22.0\%]$)
  - Ford: $-1.6\%$ ($[-14.2\%, +12.5\%]$)
  - Skoda: $-5.0\%$ ($[-18.9\%, +7.8\%]$)
  - Hyundai: $-13.1\%$ ($[-27.9\%, -1.0\%]$)
  - Toyota: $-16.6\%$ ($[-31.6\%, -5.0\%]$)
  - Vauxhall: $-19.4\%$ ($[-35.6\%, -9.0\%]$)
- **Posterior predictive checks:**
  - Overall distribution and tails (< £2k, > £50k) align closely with observed data.
  - **Per-brand bias completely eliminated:** While M1 suffered large systematic errors (+18% to -18%), M2 brings mean prediction bias within $[-0.5\%, +0.6\%]$ across all 9 brands (e.g. Mercedes error drops from $+8.9\%$ to $-0.05\%$, Vauxhall from $-18.1\%$ to $-0.01\%$, Audi from $+17.7\%$ to $-0.47\%$).
- **Prior sensitivity:**
  - Refit with Vague ($\sigma_{\text{brand}} \sim \text{HalfNormal}(5.0)$) and Informative ($\sigma_{\text{brand}} \sim \text{HalfNormal}(0.25)$) priors.
  - Posterior parameters remain stable and virtually identical across all 3 specifications: $\sigma = 0.185$ across all, $\sigma_{\text{brand}} = 0.194 \pm 0.01$ across all, slopes identical to 3 decimal places. Data likelihood dominates.

### M3: full Bayesian workflow results (2026-11-16, `notebooks/04_m3_brand_model.ipynb`)

M1 and M2 were refitted in this notebook with the same priors (their saved `.nc` files were not on this machine), so the M1/M2 numbers quoted here come from those refits. They agree with the sections above except where noted under "To check".

- **Sampling:** 4 chains, 1,000 draws, 1,000 tune, `target_accept=0.95`, seed 2026, nutpie, centred group effects, low-rank mass matrix. Runtime 39s.
- **Convergence:** R-hat 1.00 for all parameters including the 168 group intercepts, min bulk ESS 3,085, min tail ESS 2,349, zero divergences.
- **Prior predictive check:** with HalfNormal(0.5) on both group sds, 5.1% of simulated prices are outside £500–£200k (M2: 3.9%, M1: 3.0%). Tightened to **HalfNormal(0.3)**, which gives 3.7%. All M3 results below use 0.3.
- **Key posterior estimates (M2 → M3):**
  - β_age −0.121 → −0.106 (−11.4% → −10.1% per year)
  - β_mil −0.048 → −0.053 (−4.7% → −5.2% per 10k miles)
  - β_eng +0.413 → +0.262 (+51% → +30% per litre): much of the engine effect was a between-model effect
  - Manual vs automatic −0.141 → −0.095 (−13% → −9%)
  - Petrol vs diesel −0.010 → +0.069 (sign flips within a model)
  - σ_brand = 0.279 (94% HDI 0.156–0.417), σ_model = 0.254 (0.225–0.283), σ = 0.127
  - Residual sd 0.185 → 0.127: 53% less unexplained variance than M2, 68% less than M1
- **Variance decomposition (answers "brand vs model"):** of the variance left after the covariates, brand 48% (94% HDI 26–71%), model within brand 42% (23–61%), individual car 10% (6–15%). P(brand share > model share) = 0.57, so the two cannot be ranked: σ_brand rests on nine brands only.
- **Posterior predictive checks:**
  - Per-model bias (same error definition as the M2 notebook): in M2, 39 of the 90 models with n ≥ 20 are off by more than 10% on average, from +39.1% (Grandland X) to −32.0% (Ka+); city cars are over-predicted, SUVs under-predicted.
  - In M3 no model with n ≥ 5 is off by more than 10% (largest 5.3%); the sd of the per-model bias drops from 18.7% to 0.9%. Per-brand bias stays within [−0.3%, +0.6%], the same as M2.
  - Tails as in M2; the share below £2,000 is still under-predicted (0.14% observed, 0.06% replicated).
- **Prior sensitivity:** fixed effects, σ_model and σ are identical under all three prior sets. σ_brand moves mildly: 0.305 (vague, HalfNormal(5)), 0.279 (main, HalfNormal(0.3)), 0.270 (informative, HalfNormal(0.25)); brand share 51% / 48% / 46%. No brand or model effect changes by more than 0.009 log units. First place in the project where the prior is visible, but far smaller than the posterior uncertainty.

## Results

### M1: prior predictive check (2026-11-04, `notebooks/02_m1_pooled.ipynb`)

Prices simulated from the priors only (1,000 draws × training-set cars), compared with a plausible range of £500–£200k. That range comes from general knowledge, not from the data.

| | 5% | median | 95% | < £500 | > £200k |
|---|---|---|---|---|---|
| Version 1 (agreed) | £334 | £13,048 | £569,076 | 6.6% | 9.9% |
| Version 2 (revised) | £1,593 | £13,085 | £115,779 | 0.7% | 2.3% |

- **Version 1 failed.** The slope sd of 0.5 is per unit, but centred age reaches +12 and centred mileage +19, so a single draw can shift log-price by 6 or more (a factor of 400). The implied depreciation curves show 15-year-old cars worth pennies or millions.
- **Version 2 passes** and is still centred on 0, so it doesn't favour any direction.
- **Lesson for the report:** set slope priors relative to the predictor's range, not just "per unit". Revising after a prior predictive check is fine because the check never uses observed prices.

## Open questions

- PyTensor warns that it can't find BLAS (installed via pip), so sampling may be slow. If M1 sampling is slow, install a BLAS or use conda-forge.
- M4/M5: add curvature in age (quadratic or spline)?
- Model comparison (LOO for M1–M4) is postponed to a separate step once M4 is fitted. The saved fits already contain the pointwise log-likelihood.
- Full three-prior refit for every model, or `priorsense` power-scaling for all and full refits only for the final model? Ask in a TA session.

## AI use log

Required for the report (Aalto AI guidelines, points 5 and 6).

| Date | Tool | What it was used for |
|---|---|---|
| 2026-10-20 | Claude Code | Data loader (`src/data.py`) and first EDA notebook |
| 2026-11-04 | Claude Code | M1 prior notebook (`02_m1_pooled.ipynb`): priors, prior predictive check, figures. `prepare()` / `train_test()` in `src/data.py`, mileage transform check, data-structure summary, discussing prior options, cleaning proposal, this progress file |
| 2026-11-12 | Antigravity | M1 baseline model completion (`02_m1_pooled.ipynb`): MCMC sampling (NUTS via Bambi/PyMC), convergence diagnostics (R-hat, ESS, divergences, rank plots), posterior predictive checks (overall, tails, brand bias), prior sensitivity refits (vague vs. informative), Numba compilation setup for pointwise log-likelihood, updating progress checklist and results |
| 2026-11-14 | Antigravity | M2 hierarchical brand model completion (`03_m2_brand.ipynb`): varying brand intercepts `(1 \| brand)`, prior checks, NUTS MCMC sampling (4 chains, target_accept=0.95, zero divergences), pointwise log-likelihood, forest plots, posterior predictive checks demonstrating elimination of brand bias (reduced from $\pm 18\%$ to $\pm 0.5\%$), 31.9% residual variance reduction, prior sensitivity refits (vague vs. informative), updating progress log |
| 2026-11-16 | Claude Code | M3 brand + model notebook (`04_m3_brand_model.ipynb`), prior predictive check, sampler set-up (nutpie, centred, low-rank mass matrix), convergence diagnostics, fixed-effects comparison with M1 and M2, variance decomposition, posterior predictive checks (overall, tails, per-brand and per-model bias), prior sensitivity refits, findings text and summary. Refitted M1 and M2 for the comparison. Corrected the reference-level wording (Diesel / Automatic) in `PROGRESS.md` and the M1/M2 notebooks. Updated this progress file. |
