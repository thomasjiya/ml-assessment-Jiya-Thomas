# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### (a) Machine Learning Problem Formulation

**Target Variable:**
`items_sold` — the number of items sold at a given store in a given month.

**Candidate Input Features:**
- `promotion_type` — the promotion deployed that month (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points)
- `store_size` — small, medium, or large
- `location_type` — urban, semi-urban, or rural
- `competition_density` — number of competing stores nearby
- `is_weekend` — whether the transaction occurred on a weekend
- `is_festival` — whether a festival or public holiday was active
- `month` — captures seasonal shopping patterns
- `day_of_week` — captures weekly footfall patterns
- `is_month_end` — captures payday spending behaviour

**Type of ML Problem:**
This is a **supervised regression problem**. The target variable `items_sold`is a continuous numerical value, not a category, so regression is appropriate.

**Justification:**
- Supervised learning is used because we have historical labelled data — past records where the promotion type, store conditions, and resulting items sold are all known. The model learns the relationship between inputs and output from these examples.
- Regression is chosen over classification because `items_sold` is a quantity rather than a discrete label. Predicting the exact volume allows the business to rank promotions by expected output and choose the one with the highest predicted sales for each store.
- Once the regression model is trained, it can be used as a what-if simulator for each store, predict `items_sold` under each of the 5 promotion types and deploy whichever scores highest.

---

### (b) Why Items Sold is a More Reliable Target Than Revenue

**The problem with revenue as a target:**
Revenue is the product of items sold and price. Flat Discount promotions directly reduce the price of each item, which mechanically lowers revenue per transaction even if many more items are sold. A model trained to predict revenue would therefore systematically undervalue high-volume discounting promotions, not because they perform poorly, but because the
target variable itself is distorted by the promotion being evaluated.

For example, a Flat Discount promotion may generate $8,000 revenue from 200 items, while a Loyalty Points promotion generates $9,000 from 150 items.
Revenue-based evaluation would favour Loyalty Points, even though Flat Discount moved 33% more stock — which may be the actual business goal (clearing inventory, driving footfall, acquiring new customers).

**Why items sold is more reliable?:**
`items_sold` measures raw sales volume independently of price changes caused by the promotions themselves. It reflects customer response to the promotion without being mathematically contaminated by the promotion's pricing effect.
This makes it a clean, unbiased signal of promotion effectiveness.

**Broader principle — Target Leakage and Metric Alignment:**
This illustrates the principle of **target variable alignment** in ML projects.
The target must measure what the business actually wants to optimise, and it must not be structurally influenced by the input features in a way that creates circular reasoning or misleading signals. Revenue here is partly determined by
promotion type (an input feature), making it a poor target. 
A good target variable is one that the model inputs can genuinely explain, without the inputs mathematically defining the output before the model has learned anything.

---

### (c) Alternative Modelling Strategy to a Single Global Model

**The problem with one global model:**
A single global model averages the behaviour of all 50 stores together. If urban stores respond strongly to BOGO promotions while rural stores respond better to Flat Discounts, a global model will blur these differences and produce recommendations that are mediocre for every store rather than optimal for any specific one.

**Proposed strategy: Location-Stratified Models**

Train a separate model for each location type (urban, semi-urban, rural), giving three models in total. Each model is trained only on historical data from stores of that type, learning promotion response patterns specific to that customer demographic and competitive environment.

This can be extended further to **store-level models** if sufficient historical data exists per store (at least several hundred records per store is recommended).

**Justification:**
- Urban customers may be more responsive to BOGO and Category Offers due to higher disposable income and variety-seeking behaviour.
- Rural customers may respond more to Flat Discounts due to greater price sensitivity.
- Semi-urban stores may behave differently again depending on local competition.

A stratified approach captures these distinct response curves without requiring a fully separate model per store. It balances **model specificity** (learning local patterns) against **data sufficiency** (each group still has
enough training records for a reliable model).

If data per store is rich enough, a **mixed-effects model** or **store-level embeddings** inside a neural network could be considered as a more sophisticated alternative that captures both shared global patterns and individual store-level deviations simultaneously.

-------------------------------------------------------------------------------------------------------------------
## B2. Data and EDA Strategy

### (a) Joining the Four Tables

**Table descriptions and join keys:**

The four tables would be joined as follows:

- `transactions` is the **base table** — it contains one row per transaction with a `store_id` and `transaction_date`.
- Join `store_attributes` onto transactions using `store_id` as the key. This brings in `store_size`, `location_type`, and `competition_density` for each transaction.
- Join `promotion_details` using `store_id` and `month` (or a date range key) as the composite key — since each store runs one promotion per month, this attaches the `promotion_type` active at that store in that period.
- Join `calendar` using `transaction_date` as the key. This brings in `is_weekend` and `is_festival` flags for each date.

All joins are **left joins** from the transactions table outward, preserving all transaction records even if a store had no promotion that month (which would be encoded as a "no promotion" category rather than a null).

**Grain of the final dataset:**
One row = one store × one month combination.

Before modelling, transactions are **aggregated to store-month level**:
- `items_sold` → sum of all items sold at that store in that month (target)
- `is_weekend` → proportion of days in that month that were weekends
- `is_festival` → binary flag (1 if any festival occurred that month)
- All store attributes remain constant per store (no aggregation needed)
- `promotion_type` → the single promotion active that month per store

This grain makes sense because promotions are decided and evaluated at the monthly store level, which is also the level at which the business makes deployment decisions.

---
### (b) Exploratory Data Analysis (EDA)

Before modelling, I would perform the following analyses:

1.  Promotion-wise Sales Comparison (Boxplot/Bar Chart)
    - Compare average items sold across different promotion types. 
    - Insight: Identify which promotions generally perform better.
    - Impact: Helps in feature importance expectations and baseline strategy.
2. Time Series Analysis (Line Chart)
    - Plot daily/weekly sales trends across stores.
    - Insight: Detect seasonality, trends, and spikes during festivals/weekends.
    - Impact: Encourages inclusion of time-based features (month, lag variables).
3. Store Segmentation Analysis (Grouped Bar Chart)
    - Compare performance across store types (urban vs rural, high vs low competition).
    - Insight: Promotions may perform differently depending on store characteristics.
    - Impact: Suggests interaction features (e.g., promotion × store type).
4. Correlation Heatmap
    - Analyze relationships between numerical variables (footfall, store size, sales).
    - Insight: Identify strong predictors and multicollinearity.
    - Impact: Guides feature selection and model choice.
5. Distribution of Target Variable (Histogram)
    - Examine skewness and outliers in items sold.
    - Insight: Sales may be highly skewed.
    - Impact: May require log transformation or robust models.
6. Promotion Effect by Day Type (Weekend vs Weekday)
    - Compare sales uplift during weekends vs weekdays for each promotion.
    - Insight: Some promotions may work better on weekends.
    - Impact: Adds interaction features (promotion × weekend).
---
### (c) Handling Promotion Imbalance

Since 80% of transactions occur without promotions, the dataset is highly imbalanced. This can lead to:

- The model being biased toward predicting no-promotion scenarios
- Underestimating the true impact of promotions
- Poor learning of promotion-specific effects

Steps to address this:

- Resampling techniques: Oversample promotion data or undersample non-promotion data
- Stratified sampling: Ensure balanced representation during training
- Feature engineering: Include a binary “promotion applied” feature and interaction terms
- Use appropriate evaluation metrics: Such as uplift or segmented performance instead of overall accuracy
- Model choice: Use models that handle imbalance well (e.g., tree-based models).

-----------------------------------------------------------------
## B3. Model Evaluation and Deployment
### (a) Train-Test Split and Evaluation Metrics

Since the data is time-series (monthly data over three years), I would use a time-based split rather than a random split.

Train-test split approach:

- Train on the first 2–2.5 years
- Validate on a subsequent period (e.g., next 3–6 months)
- Test on the final few months
- Optionally use rolling/forward validation (walk-forward validation) to ensure robustness across time

**Why random split is inappropriate:** It would mix past and future data, causing data leakage. The model could learn patterns from the future, leading to over-optimistic performance.
It ignores temporal dependencies like seasonality and trends.

**Evaluation metrics:**

1. RMSE (Root Mean Squared Error)
    - Measures prediction error magnitude
    - Penalizes large errors more heavily
    - Interpretation: Useful to ensure the model avoids large forecasting mistakes in sales
2. MAE (Mean Absolute Error)
    - Measures average absolute error
    - More interpretable than RMSE
    - Interpretation: “On average, the model is off by X items sold per store per month”
3. R² (Coefficient of Determination)
    - Measures how much variance in sales is explained by the model
    - Interpretation: Higher R² indicates better explanatory power
4. Business Metric: Incremental Lift / Uplift
    - Measures how much additional sales a promotion generates vs no promotion
    - Interpretation: Most important metric for decision-making, as the goal is to maximize sales impact of promotions
---
### (b) Explaining Different Recommendations Using Feature Importance

The model recommends different promotions for the same store in December vs March due to changing feature values across time.

**How to investigate:**

1. Global Feature Importance
    - Identify key drivers (e.g., festival flag, seasonality, footfall, past sales, promotion type interactions)
2. Local Explanation (per prediction)
    - Use methods like SHAP values to explain why a specific prediction was made for:
        - Store 12 in December
        - Store 12 in March
3. Compare feature values across months
    - December may have: 
        - Festival/holiday season → higher demand, Higher footfall.
        - Customers more responsive to Loyalty Points Bonus
    - March may have:
        - Lower seasonal demand
        - Need for stronger incentives → Flat Discount performs better

**How to communicate to marketing team:**

1. In December, festive demand and high customer engagement make loyalty rewards more effective.
2. In March, demand is lower, so direct price reductions (discounts) drive higher conversions.”
3. Use simple visuals (bar charts of feature contributions) to make explanations intuitive

---
### (c) Deployment and Monitoring Strategy

1. Model Saving

- Save the trained model using formats like: pickle / joblib (Python)
- Store along with: Feature transformation pipeline / Model version and training metadata

2. Monthly Data Preparation
At the start of each month:

Collect latest data:
- Store attributes (if updated) like recent sales history/calendar features (festivals, weekends). 
- Apply the same feature engineering pipeline used during training. 
- Generate input dataset at store–month level

3. Generating Recommendations

- Feed the prepared data into the saved model
- Model outputs predicted sales for each promotion type
- Select the promotion with maximum predicted sales for each store

4. Deployment Pipeline

- Automate using a workflow tool (e.g., scheduled batch job)

5. Monitoring and Retraining

To detect performance degradation:

- Prediction vs Actual Tracking
- Data Drift Monitoring
- Concept Drift
- Business KPI Tracking

**Retraining triggers:**

- Significant drop in model performance
- Major change in customer behavior or market conditions
- Periodic retraining (e.g., every 6 months)

---