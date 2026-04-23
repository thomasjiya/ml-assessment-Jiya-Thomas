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

If data per store is rich enough, a **mixed-effects model** or
**store-level embeddings** inside a neural network could be considered as a more sophisticated alternative that captures both shared global patterns and individual store-level deviations simultaneously.