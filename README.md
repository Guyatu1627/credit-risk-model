# Bati Bank - Credit Risk Assessment Model

This repository houses an end-to-end Credit Scoring Model developed for Bati Bank in partnership with an eCommerce platform to roll out a Buy-Now-Pay-Later (BNPL) service.

---

## Credit Scoring Business Understanding

### 1. Basel II Accord & Model Interpretability
The Basel II Capital Accord dictates strict regulatory expectations for risk measurement, model transparency, and auditability. Because credit scoring models directly influence financial access and lending risk, "black box" models are generally unacceptable without exhaustive justification. Our model must be completely transparent, highly interpretable, and thoroughly documented so that external regulators and internal risk management teams can confidently validate how risk probabilities are calculated.

### 2. The Necessity and Risks of Proxy Target Variables
Because our eCommerce dataset lacks explicit historical "default" labels, we must engineer a proxy target variable to signify default using alternative customer behavioral patterns (Recency, Frequency, and Monetary values). 
* **Business Risks:** This proxy approach introduces the risk of *Misclassification*. True low-risk customers who simply shop infrequently could be wrongly labeled as high-risk (False Positives), blocking sales. Conversely, highly active platform users might be financially unstable and prone to actual default, exposing the bank to capital losses if the proxy fails to flag them (False Negatives).

### 3. Model Architecture Trade-offs

| Factor | Simple Interpretable Models (e.g., Logistic Regression + WoE) | High-Performance Models (e.g., Gradient Boosting / XGBoost) |
| :--- | :--- | :--- |
| **Interpretability** | **Extremely High.** Generates clean, linear credit scorecards that easily explain risk weights to stakeholders. | **Low.** Features highly non-linear relationships, acting as a complex mathematical "black box." |
| **Predictive Power** | **Moderate.** Might miss intricate, non-linear risk patterns buried in behavioral alternative data. | **Extremely High.** Excellent at capturing subtle behavioral signals to minimize credit losses. |
| **Regulatory Compliance** | **Industry Gold Standard.** Seamlessly matches Basel II documentation and transparency rules. | **Complex.** Requires extra interpretability frameworks (like SHAP or LIME) to clear regulatory audits. |

---

## Task 3: Feature Engineering & Risk Proxy Setup

### 1. Feature Extraction Methodology
Using the transactional snapshots, records were aggregated at the individual `CustomerId` level to capture continuous behavior:
* **Recency (R):** Number of days elapsed since the customer's last observed transaction transaction relative to a standardized operational baseline.
* **Frequency (F):** Count of total transactions submitted, capturing platform lifecycle adoption velocity.
* **Monetary (M):** Total monetary footprint, average processing sizing, and maximum transaction value spikes.
* **Dispersion Metrics:** Standard deviation scores calculated per user to differentiate steady utility spenders from unpredictable transaction behavior.

### 2. Credit Risk Default Proxy Design
To overcome the absence of explicit default tracking tags, a programmatic classification rule engine was implemented to categorize profiles:
* **Bad Risk (Label 1):** Customers exhibiting active platform fraud flags OR falling into the lower 25th percentiles of transactional frequency and total spending volume.
* **Good Risk (Label 0):** Stable, high-engagement users displaying reliable operational continuity, making them prime candidates for the Buy-Now-Pay-Later (BNPL) rollout.