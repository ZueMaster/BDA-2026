# BDA Project Report Plan: Hierarchical Bayesian Analysis of Used Car Depreciation

This plan structures the final report for the Bayesian Data Analysis (CS-E5710) project. It strictly aligns with the official **FeedbackFruits Project Work Report Rubric** (`report_grading.md`), ensuring that every evaluation criterion (Q1–Q28) is addressed for all models (M1, M2, and M3) to target full marks (5/5).

---

## Report Overview & Document Structure

* **Target Length:** Under 20 pages in the main body (excluding Title, Table of Contents, References, and Appendix).
* **Tone & Audience:** Accessible, cohesive Bayesian data analysis story written for peers and course staff.
* **Numerical Reporting:** Strict compliance with the course *Digits* guideline (reporting posterior summaries to 2–3 significant digits based on uncertainty, avoiding artificial precision).
* **Code in Main Text:** Only readable model formulas and primary sampling invocations; extensive scripts and raw diagnostic outputs belong in the Appendix.

---

## Detailed Section-by-Section Plan

### Header & Metadata
* **Title:** *Hierarchical Bayesian Analysis of Used Car Depreciation: Decomposing Price Variation Across Brands and Models*
* **Authors:** Group members' names, student numbers, and contact details.
* **Repository:** Clickable link to the GitHub repository.

---

### 1. Introduction *(Addresses Q2, Q3, Q27 — Target: Score 5)*
* **1.1 Motivation & Background:**
  * Importance of used car valuation for consumers, dealerships, and insurance risk assessment.
  * Real-world complexity: Vehicles are not commodities; prices depend on physical wear (age, mileage, engine displacement) and hierarchical brand prestige.
* **1.2 Research Questions:**
  1. *Depreciation Rate:* How rapidly do vehicles lose value with age and mileage when holding other characteristics fixed?
  2. *Variance Decomposition:* How much of the unexplained price variation is attributable to the manufacturer (brand) versus the specific vehicle model within that brand?
* **1.3 Bayesian Workflow & Narrative:**
  * Outline the iterative modeling journey:
    $$\text{M1 (Pooled Baseline)} \xrightarrow{\text{PPC reveals brand bias}} \text{M2 (Brand Hierarchy)} \xrightarrow{\text{PPC reveals model bias}} \text{M3 (Nested Hierarchy)}$$

---

### 2. Description of the Data and Problem *(Addresses Q4, Q5 — Target: Score 5)*
* **2.1 Origin & Relation to Existing Analyses:**
  * UK used car dataset (~97k rows, 2020 listings scraped from Kaggle across 9 major manufacturers).
  * Contrast with standard non-Bayesian analyses (OLS, gradient boosting) by emphasizing Bayesian uncertainty quantification, explicit prior elicitation, hierarchical partial pooling, and posterior predictive distributions.
* **2.2 Cleaning Rules & Structure:**
  * Clear justification for cleaning choices: restricting age $\le 15$ years (classic car pricing dynamics differ), dropping rare/missing categories ($<1\%$ of data), retaining rare car models to demonstrate Bayesian shrinkage.
  * Structural hierarchy: listing $\to$ model ($J_{\text{model}} = 159$ in train) $\to$ brand ($J_{\text{brand}} = 9$).
* **2.3 Variable Transformations & Sample Split:**
  * Log-price ($\log(\text{price})$): Justified by positive skewness and multiplicative percentage depreciation.
  * Mileage representation: Justification for linear scale in 10,000-mile units based on residual analysis (avoiding high leverage and bias of $\log(\text{mileage})$ near zero).
  * Predictor centering: Centering continuous predictors ($\text{age}_c$, $\text{mileage}_{10k,c}$, $\text{engineSize}_c$) so that the intercept represents a typical vehicle.
  * Train/Test Split: 10,000-row representative training sample and a disjoint 10,000-row held-out test sample (including 9 unseen car models to test out-of-sample group generalization).

---

### 3. Description of the Models *(Addresses Q6, Q7, Q10 — Target: Score 5)*
* **3.1 Model 1 ($M_1$) — Pooled Baseline Linear Regression:**
  * Mathematical formulation:
    $$\log(\text{price}_i) \sim \text{Normal}(\mu_i, \sigma)$$
    $$\mu_i = \alpha + \beta_{\text{age}}\,\text{age}_{c,i} + \beta_{\text{mil}}\,\text{mil}_{c,i} + \beta_{\text{eng}}\,\text{eng}_{c,i} + \mathbf{x}_{\text{cat},i}^\top \boldsymbol{\beta}_{\text{cat}}$$
  * Serves as the non-hierarchical benchmark ignoring group clustering.
* **3.2 Model 2 ($M_2$) — Hierarchical Brand Varying Intercepts:**
  * Mathematical formulation:
    $$\mu_i = \alpha + \alpha_{\text{brand}[i]} + \mathbf{x}_i^\top \boldsymbol{\beta}, \quad \alpha_{\text{brand}} \sim \text{Normal}(0, \sigma_{\text{brand}})$$
  * Evaluates manufacturer-level price premiums while assuming models within a brand are identical.
* **3.3 Model 3 ($M_3$) — Nested Hierarchical Brand & Model Intercepts:**
  * Mathematical formulation:
    $$\mu_i = \alpha + \alpha_{\text{brand}[i]} + \gamma_{\text{model}[i]} + \mathbf{x}_i^\top \boldsymbol{\beta}$$
    $$\alpha_{\text{brand}} \sim \text{Normal}(0, \sigma_{\text{brand}}), \quad \gamma_{\text{model}} \sim \text{Normal}(0, \sigma_{\text{model}})$$
  * Decomposes variance across the full nested hierarchy and shares statistical strength across rare and common models via partial pooling.

---

### 4. Prior Specifications and Justifications *(Addresses Q8, Q9 — Target: Score 5)*
* **4.1 Prior Justifications & Domain Background:**
  * Baseline intercept $\alpha \sim \text{Normal}(9.5, 1.0)$: Centers typical vehicle log-price at $\approx £13,400$.
  * Slopes $\beta_{\text{age}}, \beta_{\text{mil}} \sim \text{Normal}(0, 0.10)$: Grounded in automotive domain expectations of roughly $\pm 10\%$ change per year / 10k miles.
  * Engine size $\beta_{\text{eng}} \sim \text{Normal}(0, 0.30)$ and categorical coefficients $\sim \text{Normal}(0, 0.50)$.
  * Residual error $\sigma \sim \text{HalfNormal}(0.50)$.
  * Group scales $\sigma_{\text{brand}}, \sigma_{\text{model}} \sim \text{HalfNormal}(0.30)$ (tightened from $0.50$ based on prior predictive checks).
* **4.2 Prior Predictive Checks:**
  * Detail the failure of naive unregularized slope priors ($\text{Normal}(0, 0.5)$ produced simulated prices from pennies to millions due to predictor ranges).
  * Validate that the revised weakly informative priors yield realistic price bounds ($<4\%$ in extreme tails) without leaking data from the response variable.

---

### 5. Computational and Inference Details *(Addresses Q10, Q11 — Target: Score 5)*
* **5.1 Model Implementation Code:**
  * Clean, readable Bambi formulas:
    * $M_1$: `log_price ~ age_c + mileage_10k_c + engineSize_c + fuelType + transmission`
    * $M_2$: `... + (1 | brand)`
    * $M_3$: `... + (1 | brand) + (1 | model)`
* **5.2 Sampler Configurations & Settings:**
  * Sampler: NUTS / HMC engine (using `nutpie` with compiled Rust/Numba backend for fast sampling).
  * Explicit settings: 4 chains, 1,000 warmup/tuning draws, 1,000 posterior draws per chain (4,000 total draws), `target_accept=0.95`, fixed seed 2026.
  * Computational feasibility: Execution times reported (~39s for $M_3$).

---

### 6. Convergence Diagnostics *(Addresses Q12, Q13, Q14 — Target: Score 5)*
> *Note: Reported and interpreted explicitly for **all three models** ($M_1$, $M_2$, and $M_3$).*

* **6.1 Potential Scale Reduction Factor ($\widehat{R}$ — Q12):**
  * Values: $\widehat{R} \le 1.01$ for all parameters in all models (all $1.00$ in $M_1$ and $M_3$, max $1.01$ in $M_2$).
  * Interpretation: Chains have converged to the same stationary target posterior distribution.
* **6.2 HMC-Specific Diagnostics (Divergences & Energy — Q13):**
  * Values: **0 divergent transitions** across all models.
  * Treedepth & E-BFMI checks: No chains saturated maximum tree depth; good energy overlap.
  * Interpretation: Confirms that the geometry of the posterior distribution (including 168 group intercepts) was traversed without pathological curvature.
* **6.3 Effective Sample Size (Bulk & Tail ESS — Q14):**
  * Values: Bulk ESS > 700 (min 736 in $M_2$, >3,000 in $M_3$) and Tail ESS > 1,200 across all parameters.
  * Interpretation: Assures low standard error of the posterior mean and accurate computation of $94\%$ HDIs.

---

### 7. Posterior Predictive Checks *(Addresses Q15 — Target: Score 5)*
> *Note: Reported and interpreted explicitly for **all three models**.*

* **7.1 Model 1 ($M_1$):**
  * Matches overall mode, but per-group PPC reveals severe brand bias ($\pm 20\%$, e.g., Vauxhall overpredicted by $-20.8\%$, Mercedes underpredicted by $+20.3\%$). Motivates $M_2$.
* **7.2 Model 2 ($M_2$):**
  * Eliminates brand bias (mean brand errors within $[-0.5\%, +0.6\%]$).
  * However, model-level PPC reveals severe within-brand misspecification (39 of 90 models off by $>10\%$). Motivates $M_3$.
* **7.3 Model 3 ($M_3$):**
  * Eliminates model-level bias (SD of per-model error drops from $18.7\%$ down to $0.9\%$; zero models with $n \ge 5$ have $>10\%$ error).
  * Tail checks: Evaluates predictive accuracy in extreme price tails ($<£2\text{k}$ and $>£50\text{k}$).

---

### 8. Prior Sensitivity Analysis *(Addresses Q18 — Target: Score 5)*
> *Note: Reported and interpreted explicitly for **all three models**.*

* **8.1 Compared Prior Sets:**
  * Weakly Informative (Main), Vague ($\sigma_{\text{group}} \sim \text{HalfNormal}(5.0)$), and Informative ($\sigma_{\text{group}} \sim \text{HalfNormal}(0.25)$).
* **8.2 Sensitivity Findings:**
  * Fixed effects ($\beta_{\text{age}}, \beta_{\text{mil}}, \beta_{\text{eng}}$) and $\sigma_{\text{model}}$ are identical to three decimal places across all prior sets (dominated by $N = 10,000$ data points).
  * $\sigma_{\text{brand}}$ exhibits slight prior sensitivity ($0.305$ Vague $\to 0.279$ Main $\to 0.270$ Informative) due to having only $J = 9$ brand clusters.
  * Interpretation: Proves that core conclusions and variance proportions are highly stable and robust.

---

### 9. Model Comparison *(Addresses Q16 — Target: Score 5)*
* **9.1 PSIS-LOO-CV Comparison Table:**
  * Comprehensive table from `az.compare`: ELPD, $\Delta\text{elpd}$, $\text{SE}(\Delta\text{elpd})$, $p_{\text{loo}}$, and Akaike weights.
  * Pareto $k$ diagnostic confirmation: All $k < 0.7$, validating PSIS-LOO estimates.
  * Statistical interpretation: $M_3$ decisively outperforms $M_2$ ($\Delta\text{elpd} \gg 2 \cdot \text{SE}$), which decisively outperforms $M_1$.
* **9.2 Practical Predictive Performance (10k Held-Out Test Set):**
  * Out-of-sample RMSE and MAE in £ across $M_1$, $M_2$, and $M_3$.
  * Practical assessment of out-of-sample group prediction on the 9 unseen car models in the test set.
* **9.3 Final Model Choice:**
  * Justify selecting $M_3$ as the preferred model based on predictive superiority, elimination of systematic biases, and alignment with research objectives.

---

### 10. Conclusion, Discussion, and Self-Reflection *(Combines Q19, Q20, Q21, Q22 — Target: Score 5)*
* **10.1 Key Findings & Answers to Research Questions (Q20, Q21):**
  * *Depreciation Rate:* Controlled annual depreciation is $-10.1\%$ per year ($\beta_{\text{age}} = -0.106$) and $-5.2\%$ per 10,000 miles ($\beta_{\text{mil}} = -0.053$).
  * *Variance Decomposition:* Unexplained price variation splits into **$48\%$ Brand** (94% HDI [26%, 71%]), **$42\%$ Model** (94% HDI [23%, 61%]), and **$10\%$ Individual car** (94% HDI [6%, 15%]). Brand and model are co-equal determinants of price ($P(\text{brand} > \text{model}) = 0.57$).
  * *Confounding Insights:* Engine size effect drops from $+51\%$ to $+30\%$ per litre when accounting for model nesting; petrol commands $+7\%$ over diesel within the same car model.
* **10.2 Limitations and Potential Improvements (Q19):**
  * *Cross-sectional nature:* 2020 cross-sectional data blends true aging with macroeconomic cohort effects.
  * *Unobserved variables:* Missing trim packages, vehicle condition, and geographic location.
  * *Linearity assumption:* Assumes constant percentage depreciation across vehicle lifespan.
  * *Future model extensions ($M_4$):* Discussing varying depreciation slopes ($1 + \text{age} \mid \text{brand}$) and non-linear splines as natural next steps.
* **10.3 Self-Reflection on What the Group Learned (Q22):**
  * Insights gained from the iterative Bayesian workflow: observing how PPCs expose misspecifications that aggregate summary statistics hide.
  * Practical understanding of prior predictive simulation and the critical importance of scaling priors to centered predictor ranges.
  * Lessons learned in managing MCMC convergence, multi-level group structures, and efficient sampling in PyMC/Bambi.

---

### References & Appendix *(Addresses Q10, Q25, Q26)*
* **References:** Academic papers, BDA course materials, and dataset sources.
* **Appendix A:** Data cleaning scripts and verification checks.
* **Appendix B:** Full parameter summary tables (means, SDs, 94% HDIs for all 168 group intercepts).
* **Appendix C:** Rank plots and trace plots for group-level variance parameters.
