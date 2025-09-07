1) Drop tenure==0 records (11 rows)

Decision: Exclude customers with tenure == 0 and blank TotalCharges.
Why: They’re not billable yet; their churn behavior isn’t informative for next-cycle risk. Keeping them distorts early-tenure patterns and creates label noise.
Alternatives: Impute TotalCharges=0 and keep them, rejected because they don’t represent actionable, at-risk accounts.

2) Keep tenure and MonthlyCharges; transform (don’t keep) raw TotalCharges

Decision: Keep tenure + MonthlyCharges; use avg_monthly_spend = TotalCharges / tenure; drop raw TotalCharges.
Why: Raw TotalCharges is collinear and noisy (pro-rating, plan changes). avg_monthly_spend preserves spend level without multicollinearity; tenure and current monthly price carry distinct signals.
Alternatives: Keep all three; or drop MonthlyCharges/tenure because “they multiply”, rejected due to performance and identifiability.

3) Preserve tri-level add-ons (Yes / No / No internet service)

Decision: Treat the add-on fields (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies) as three states, not binary.
Why: “No internet service” has different churn behavior from “No.” Collapsing them loses signal.
Alternatives: Collapse to binary, faster to encode but empirically worse separation.

4) Drop PhoneService and MultipleLines (and gender)

Decision: Remove these outright.
Why: Empirically weak predictors on this dataset; MultipleLines is logically dependent on PhoneService. Dropping cuts noise and dimensionality.
Alternatives: Keep and let regularization handle it, unnecessary complexity for negligible gain.

5) No “total services” roll-up; use targeted counts instead

Decision: Skip a broad total_services feature. Add streaming_count (0–2) and security_support_count (0–4) instead.
Why: These capture two distinct adoption patterns (entertainment vs protection/support) with less redundancy and clearer interpretation.
Alternatives: Use a single total count, rejected as it blurs segments and correlates with both counts.

6) Don’t bucket continuous features for trees

Decision: Keep tenure, MonthlyCharges, avg_monthly_spend as continuous for tree models.
Why: Trees split continuous variables natively; bucketing discards signal.
Alternatives: Global bucketing for all models, kept as an optional, interpretability add-on for linear models only.

7) Choose PR-AUC as the primary model selection metric

Decision: Optimize/compare models primarily on PR-AUC (churn=positive); report ROC-AUC and F1 too.
Why: Class imbalance (~26.5% churn) makes PR-AUC more sensitive to improvements where it matters (precision/recall on the positive class).
Alternatives: Accuracy/ROC-AUC only—risk of selecting models that look good but don’t find churners efficiently.

8) Handle imbalance with class weights + threshold tuning (not SMOTE baseline)

Decision: Use class_weight="balanced" and tune the decision threshold on validation to maximize F1; apply that threshold to test.
Why: Keeps the feature space intact, avoids synthetic examples that can distort categorical mixes, and aligns the operating point with business trade-offs.
Alternatives: SMOTE/undersampling as default, reserved for later if class weights + thresholding underperform.

9) Model lineup: LR (linear), RF (tree), SVM-RBF (other); evaluate CatBoost

Decision: Start with Logistic Regression, Random Forest, SVM-RBF for breadth; add CatBoost because gradient boosting on tabular + categoricals typically wins.
Why: LR gives a strong interpretable baseline; RF/SVM cover non-linearities; CatBoost is state-of-the-art for this problem family.
Alternatives: XGBoost/LightGBM instead of CatBoost, viable; we prioritized CatBoost for native categorical handling and competitive accuracy.

10) Feature importance via permutation (model-agnostic) + model-specific views

Decision: Use permutation importance (AP scorer) for the authoritative ranking; supplement with LR coefficients and RF impurity importances.
Why: Model-agnostic importance is harder to game and directly reflects impact on the target metric; model-specific views aid interpretation.
Alternatives: Rely on impurity importances only, can be biased toward high-cardinality OHE features.

11) Select by validation PR-AUC, then tune only the winner

Decision: Pick the best family on validation PR-AUC; run RandomizedSearchCV for that family; retrain on train+val; evaluate once on test.
Why: Controls overfitting risk, keeps the work scoped, and preserves a true holdout.
Alternatives: Full grid across all models, costly and unnecessary for this challenge.

12) Ship CatBoost as primary, keep LR as fallback

Decision: Production recommendation: CatBoost as the main scorer (better PR-AUC), LR as a lightweight fallback path.
Why: CatBoost gives consistent lift; LR is cheap, robust, and easy to explain if you need a rapid rollback or auditability.
Alternatives: Ensemble/stacking, possible next step, but beyond the required scope.