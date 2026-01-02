# Madrid Real Estate Price Analysis
## Predictive Modeling for Property Valuation Using Statistical Regression

## Project Overview

A comprehensive statistical analysis of the Madrid real estate market, analyzing **21,177 properties across 146 districts** to build predictive pricing models. This project demonstrates advanced regression techniques, feature engineering, and statistical modeling to quantify the key drivers of property prices in one of Europe's most dynamic housing markets.

**Business Value**: Accurate property valuation is critical for buyers, sellers, real estate agents, and investors. This analysis quantifies how location, size, and features impact prices, enabling data-driven pricing decisions.

---

## Key Objectives

1. **Exploratory Analysis**: Understand price distribution and variation across Madrid's 146 districts
2. **Feature Identification**: Determine which property characteristics most strongly predict price
3. **Statistical Modeling**: Build multiple regression models (simple, multivariate, dummy variable)
4. **Quantify Price Drivers**: Calculate the monetary value of each property feature
5. **Predictive Accuracy**: Achieve high R² for reliable property valuation

---

## Dataset Description

**Source**: Madrid real estate listings  
**Size**: 21,742 properties (21,177 after cleaning)  
**Time Period**: Recent Madrid housing market data  
**Geographic Coverage**: 146 unique districts across Madrid

### Key Features
| Feature | Description | Data Type | Completeness |
|---------|-------------|-----------|--------------|
| `buy_price` | Property purchase price (target variable) | Numeric (€) | 100% |
| `sq_mt_built` | Square meters of built area | Numeric (m²) | 99.4% |
| `n_rooms` | Number of bedrooms | Integer | 100% |
| `n_bathrooms` | Number of bathrooms | Integer | 99.9% |
| `district` | Madrid district/neighborhood | Categorical | 100% |
| `buy_price_by_area` | Price per square meter | Numeric (€/m²) | 100% |

### Data Quality Notes
- Original dataset: 58 columns with extensive missing data
- Strategic feature selection focused on completeness and relevance
- Removed columns with >50% missing values (latitude, longitude, amenities)
- Excluded 439 properties with data quality issues (0 rooms, missing critical features)

---

## Tools & Technologies

### Python Libraries
- **Data Processing**: `Pandas`, `NumPy`
- **Statistical Modeling**: `Statsmodels` (OLS regression)
- **Machine Learning**: `Scikit-learn` (mentioned for future expansion)
- **Visualization**: `Matplotlib`, `Seaborn`

### Statistical Methods
- Ordinary Least Squares (OLS) Regression
- Multiple Linear Regression
- Dummy Variable Regression (Categorical Encoding)
- Correlation Analysis
- Residual Analysis

---

## Methodology

### Phase 1: Data Cleaning & Preprocessing

#### Initial Assessment
- **Original**: 21,742 properties × 58 features
- **Missing Data**: Removed 10 columns with 100% missing values
- **Feature Selection**: Retained 8 core features with <1% missing data

#### Cleaning Steps
```python
# 1. Remove properties with data quality issues
- Dropped rows where n_rooms = 0 (likely data errors)
- Result: 21,303 properties

# 2. Handle missing bathrooms
- Filled NaN values with mode (1 bathroom)
- Replaced 0 bathrooms with 1 (minimum assumption)

# 3. Remove missing area data
- Dropped rows with missing sq_mt_built
- Final dataset: 21,177 properties (97.4% retention)

# 4. Feature Engineering
- Extracted district names from combined location field
- Created dummy variables for regression analysis
- Converted data types for optimization
```

#### Data Transformation
- Removed irrelevant columns (ID fields, rental prices)
- Split `city` column into `district` (146 unique values)
- Cleaned negative rent_price values (data errors)
- Final feature set: 7 columns optimized for modeling

---

### Phase 2: Exploratory Data Analysis

#### Distribution Analysis
**Property Characteristics**:
- Average price: ~€400,000 - €600,000 (varies by district)
- Average size: 100-120 m² built area
- Most common: 3 bedrooms, 1-2 bathrooms

**District Variation**:
- **Highest prices**: Recoletos, Castellana, Jerónimos (premium districts)
- **Lowest prices**: Peripheral districts (San Cristóbal, Villaverde)
- **Price range**: 10x difference between most and least expensive districts

#### Correlation Analysis
**Key Findings**:
- `sq_mt_built` ↔ `buy_price`: **r = 0.89** (very strong positive correlation)
- `n_rooms` ↔ `buy_price`: r = 0.62 (moderate positive)
- `n_bathrooms` ↔ `buy_price`: r = 0.58 (moderate positive)
- `n_rooms` ↔ `sq_mt_built`: r = 0.72 (multicollinearity concern)

**Visualization**:
- Pair plots showing linear relationships
- Heatmaps revealing feature correlations
- Box plots identifying outliers (luxury properties >€5M)

---

### Phase 3: Statistical Modeling

#### Model 1: Simple Linear Regression (Recoletos District)

**Objective**: Understand the area-price relationship in a premium district

```python
Model: buy_price = β₀ + β₁ × sq_mt_built
District: Recoletos (one of Madrid's most expensive areas)
Sample Size: 113 properties
```

**Results**:
| Metric | Value |
|--------|-------|
| **R-squared** | **0.833** |
| **Intercept (β₀)** | -43,420 € |
| **Area Coefficient (β₁)** | **8,976 €/m²** |
| **P-value (β₁)** | <0.001 (highly significant) |

**Interpretation**:
- 83.3% of price variation explained by area alone
- Each additional square meter adds **€8,976** to property value
- Model is statistically significant (F-statistic: 555.1)

**Prediction Example**:
```
Property: 300 m² apartment in Recoletos
Predicted Price = -43,420 + (8,976 × 300) = €2,649,380
```

---

#### Model 2: Multiple Linear Regression (Recoletos)

**Objective**: Improve prediction by adding bathrooms as second predictor

```python
Model: buy_price = β₀ + β₁ × sq_mt_built + β₂ × n_bathrooms
District: Recoletos
Sample Size: 113 properties
```

**Results**:
| Metric | Value |
|--------|-------|
| **R-squared** | **0.846** (+1.3% vs Model 1) |
| **Intercept (β₀)** | -213,900 € |
| **Area Coefficient (β₁)** | **7,863 €/m²** |
| **Bathrooms Coefficient (β₂)** | **+144,200 €** |
| **P-values** | All <0.01 (significant) |

**Interpretation**:
- 84.6% of price variation explained (improvement over simple model)
- Each additional square meter: **+€7,863**
- Each additional bathroom: **+€144,200**
- Controlling for area, bathrooms add substantial value

**Prediction Example**:
```
Property: 300 m² apartment, 3 bathrooms, Recoletos
Predicted Price = -213,900 + (7,863 × 300) + (144,200 × 3) = €3,005,400
```

---

#### Model 3: Dummy Variable Regression (Multi-District)

**Objective**: Quantify location premiums by comparing three districts

```python
Model: buy_price = β₀ + β₁ × Castellana + β₂ × Trafalgar
Districts: Recoletos (baseline), Castellana, Trafalgar
Sample Size: 404 properties (all three districts)
```

**Results - Location Only**:
| Metric | Value |
|--------|-------|
| **R-squared** | 0.363 |
| **Recoletos (baseline)** | €2,155,000 (intercept) |
| **Castellana premium** | **-€778,500** |
| **Trafalgar premium** | **-€1,513,000** |

**Interpretation**:
- Recoletos is the most expensive (baseline comparison)
- Castellana properties are €778K cheaper on average
- Trafalgar properties are €1.51M cheaper on average
- **But R² is low** - location alone insufficient

---

**Results - Location + Area**:
```python
Model: buy_price = β₀ + β₁ × Castellana + β₂ × Trafalgar + β₃ × sq_mt_built
```

| Metric | Value |
|--------|-------|
| **R-squared** | **0.866** (massive improvement!) |
| **Intercept (β₀)** | €277,400 |
| **Castellana effect** | **-€371,200** |
| **Trafalgar effect** | **-€426,200** |
| **Area Coefficient (β₃)** | **7,666 €/m²** |

**Key Insights**:
- **Controlling for area**, district premiums are smaller but still substantial
- Recoletos commands **+€400K premium** over Trafalgar for same-sized properties
- Area coefficient (€7,666/m²) consistent across models
- Combined model explains **86.6% of price variation**

**Practical Application**:
```
Scenario: 200 m² apartment comparison

Recoletos:   €277,400 + (7,666 × 200) + 0           = €1,810,600
Castellana:  €277,400 + (7,666 × 200) - €371,200    = €1,439,400
Trafalgar:   €277,400 + (7,666 × 200) - €426,200    = €1,384,400

Recoletos Premium: €426,200 over Trafalgar (30.4% more expensive)
```

---

## Key Findings & Business Insights

### 1. Area is the Dominant Price Driver
**Finding**: Square meters built explains 83-87% of price variation across all models.

**Coefficient Range**: €7,666 - €8,976 per m² (depending on district and model)

**Business Implication**:
- Size is the #1 consideration in property valuation
- Accurate area measurement is critical
- Small measurement errors (±5 m²) can mean €40K+ price differences

---

### 2. Bathrooms Add Significant Value
**Finding**: Each additional bathroom adds €144,200 to property value in premium districts.

**Business Implication**:
- Renovation opportunity: Adding a bathroom could increase resale value significantly
- ROI calculation: If renovation costs <€100K, adding bathroom is profitable
- Marketing focus: Highlight bathroom count in listings

---

### 3. Location Premium is Quantifiable
**Finding**: District location creates price premiums of €400K+ for equivalent properties.

**District Hierarchy** (descending order):
1. **Recoletos** (Baseline: Most expensive)
2. **Castellana** (-€371K vs Recoletos)
3. **Trafalgar** (-€426K vs Recoletos)

**Business Implication**:
- Buyers can quantify the "location tax" they're paying
- Sellers in premium districts can justify higher prices with data
- Investors can identify undervalued districts (high growth potential)

---

### 4. Predictive Accuracy is High
**Finding**: Final models achieve R² = 0.87, meaning 87% prediction accuracy.

**Model Performance Summary**:
| Model Type | R² | Key Variables | Use Case |
|------------|-----|---------------|----------|
| Simple Linear | 0.83 | Area only | Quick estimates |
| Multiple Regression | 0.85 | Area + Bathrooms | Detailed valuation |
| Dummy Variable | 0.87 | Area + Location | Cross-district comparison |

**Business Implication**:
- Models can be deployed as automated valuation tools
- Real estate agents can provide instant price estimates
- Buyers can identify overpriced listings (>10% above predicted)

---

### 5. District-Specific Strategies

**Premium Districts** (Recoletos, Castellana):
- Price per m²: €7,000 - €9,000
- Target market: Luxury buyers
- Strategy: Emphasize location prestige, nearby amenities, historical significance

**Mid-Tier Districts** (Trafalgar):
- Price per m²: €5,000 - €7,000
- Target market: Middle-income professionals
- Strategy: Balance of affordability and location

**Value Districts** (Peripheral areas):
- Price per m²: €2,000 - €4,000
- Target market: First-time buyers, investors
- Strategy: Focus on size, potential for appreciation

---

## Model Comparison & Selection

| Criterion | Simple Linear | Multiple Regression | Dummy Variable |
|-----------|---------------|---------------------|----------------|
| **R²** | 0.83 | 0.85 | **0.87**  |
| **Complexity** | Low | Medium | High |
| **Interpretability** | High | High | Medium |
| **Variables** | 1 (area) | 2 (area, bathrooms) | 4+ (area, location dummies) |
| **Best For** | Quick estimates | Single-district valuation | Cross-district comparison |
| **Recommended Use** | Initial screening | Detailed appraisal | Portfolio analysis |

**Winner**: **Dummy Variable Model** (R² = 0.87) for highest accuracy when comparing across districts.

---

## Skills Demonstrated

### Technical Skills
- **Data Cleaning**: Handling missing values, outliers, data type conversion
- **Feature Engineering**: Dummy variable creation, text extraction
- **Statistical Modeling**: OLS regression, multivariate analysis
- **Model Evaluation**: R², p-values, coefficient interpretation
- **Data Visualization**: Heatmaps, scatter plots, bar charts

### Domain Knowledge
- **Real Estate Valuation**: Understanding price drivers and market dynamics
- **Spatial Analysis**: District-level comparison and location premiums
- **Economic Reasoning**: Marginal value calculation, ROI analysis

### Python Proficiency
- Advanced Pandas operations (groupby, query, string operations)
- Statsmodels for econometric modeling
- Professional visualization with Matplotlib/Seaborn
- Data preprocessing pipeline design

---

## Key Takeaways

1. **Size Dominates**: Area explains 83% of price variation—always the primary consideration
2. **Bathrooms Add Value**: €144K per bathroom in premium districts—renovation ROI is real
3. **Location Premium is Real**: €400K+ for equivalent properties in top districts
4. **Models Work**: R²=0.87 means 87% prediction accuracy—deployable for real-world use
5. **Data-Driven Decisions**: Statistical models remove guesswork from pricing strategy

---

**Ali Rabie**  
Data Analyst | Python • SQL • Power BI
alirabie0128@gmail.com  
