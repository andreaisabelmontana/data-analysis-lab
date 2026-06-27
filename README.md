# data-analysis-lab

Interactive companion for *Fundamentals of Data Analysis* (BCSAI, IE University). 12 self-contained visual demos of statistical inference, plus a tested library of the numerical cores behind them.

**Live:** https://andreaisabelmontana.github.io/data-analysis-lab/

## The demos

Module 1 — Sampling & estimation
1. Sampling distribution of the mean & the CLT — repeated samples build the sampling distribution of X̄, which narrows as `SE = σ/√n` and turns Normal whatever the population shape
2. Maximum likelihood estimation — slide a parameter along the log-likelihood and find the MLE (exponential rate or Normal mean)
3. Method of moments vs. MLE — both estimators on the same sample, with replicate RMSE

Module 2 — Confidence intervals
4. CI coverage — many 95% intervals; red ones miss μ; observed coverage approaches the stated level
5. CI width vs. n & confidence — half-width `t · s/√n` shrinks like 1/√n

Module 3 — Hypothesis testing (one sample)
6. One-sample z / t test — statistic against a shaded rejection region
7. The p-value as a tail area
8. Type I / II error & power — α, β and power = 1−β

Module 4 — Two-sample inference
9. Two-sample t-test — pooled vs. Welch (Satterthwaite df)
10. Paired vs. unpaired

Module 5 — ANOVA & categorical data
11. One-way ANOVA — `F = MSB / MSE`
12. Chi-square test of independence — `E = Rᵢ·Cⱼ/N`, `χ² = Σ(O−E)²/E`

Plain HTML + canvas + KaTeX, no charting or stats libraries, zero build step. The page (`index.html`) loads `src/app.js` as an ES module, which imports the cores below.

## Tested numerical cores (`src/lib`)

The math is extracted into framework-free ES modules (no DOM, no globals) and unit-tested with Node's built-in runner. `src/app.js` imports the distribution tails, the RNG and the sample summaries from here, so the page and the tests share one implementation.

| Module | Functions |
| --- | --- |
| `distributions.js` | `normPdf` `normCdf` `normInv` · `tPdf` `tCdf` `tTailTwo` `tInv` · `chiSqTail` · `fTail` · `logGamma` `gammaP` `betaI` (A&S erf, Lanczos log-gamma, continued-fraction incomplete gamma/beta) |
| `random.js` | `mulberry32`/`makeRng` (seedable) · `randn` (Box-Muller) · `randExp` · `randGamma` (Marsaglia-Tsang) |
| `summary.js` | `mean` `variance` `std` · `median` `mode` `quantile` `quartiles` `iqr` · `outliers`/`outlierBounds` (1.5·IQR) · `summarize` |
| `correlation.js` | `covariance` · `correlation` (Pearson) |
| `regression.js` | `linearRegression` (OLS slope/intercept + R²) · `predict` |
| `transform.js` | `minMax` · `zScore` · `binIndex` · `histogram` |
| `groupby.js` | `groupBy` · `aggregate` · `sumBy` `meanBy` `countBy` |
| `inference.js` | `ciHalfWidth` · `oneSampleTest` · `twoSampleTest` (pooled+Welch) · `pairedTest` · `anova` · `chiSquareIndependence` |

Quantiles use linear interpolation between order statistics (R-7 / NumPy default); variance/covariance use the n−1 divisor.

### Properties proven by the tests

- summary stats (mean, sample variance/std, median, quartiles, IQR) match hand-computed values; the 1.5·IQR rule flags a known outlier and nothing else
- Pearson correlation is **+1 / −1** for perfectly (anti-)linear data; OLS recovers a known slope and intercept with **R² = 1** on exact linear data
- min-max scaling lands in **[0, 1]**; z-score gives **zero mean, unit variance**; binning assigns each value to the correct equal-width bin
- group-by sum / mean / count are correct on a constructed dataset
- distribution tails match textbook quantiles (`normCdf(1.96)≈0.975`, `t_{0.975,10}≈2.228`, `χ²_{0.95,1}≈3.841`)
- the seeded RNG (`makeRng`) is deterministic and reproducible across calls

## Run

Open `index.html` in any browser, or serve the folder statically. No dependencies, no build.

## Test

Node 24+, no npm install (uses `node:test` + `node:assert` only):

```
node --test
```

```
✔ correlation is +1 for perfectly linear data
✔ correlation is -1 for perfectly anti-linear data
✔ correlation stays within [-1, 1] for noisy data
✔ covariance has the expected sign and value
✔ regression recovers a known slope, intercept and R^2 = 1
✔ R^2 drops below 1 when residuals are nonzero
✔ standard normal pdf and cdf hit known points
✔ normInv inverts normCdf
✔ two-sided t tail matches the normal in the large-df limit
✔ tInv returns the textbook 95% critical value
✔ chi-square upper tail at known quantiles
✔ F upper tail is a probability in (0,1)
✔ groupBy partitions rows by key
✔ sum aggregation totals each group correctly
✔ mean aggregation averages each group correctly
✔ count aggregation counts members per group
✔ aggregate works with min/max reducers
✔ CI half-width = t * s / sqrt(n)
✔ one-sample z-test rejects a clear effect
✔ one-sample t-test fails to reject a tiny effect
✔ two-sample pooled and Welch agree when sizes and variances match
✔ paired t-test on a consistent shift is significant
✔ ANOVA F is large when group means are far apart
✔ ANOVA on identical groups gives F near 0 and large p
✔ chi-square independence: associated table rejects, uniform does not
✔ seeded RNG is deterministic and reproducible
✔ seeded randn has roughly zero mean over many draws
✔ mean matches hand value
✔ sample variance and std use the n-1 divisor
✔ median is the R-7 interpolated 50th percentile
✔ quartiles match hand-computed R-7 values
✔ quantile endpoints return the extremes
✔ mode returns the most frequent value
✔ 1.5*IQR rule flags a known outlier and nothing else
✔ no outliers in tight, symmetric data
✔ summarize bundles the descriptive set
✔ min-max scales into [0,1] with 0 and 1 attained
✔ min-max maps a constant series to zeros
✔ z-score yields zero mean and unit sample variance
✔ binIndex buckets values into the correct equal-width bin
✔ histogram counts assign every value to a bin
ℹ tests 41
ℹ pass 41
ℹ fail 0
```

Part of the *-lab series: [discrete-math-lab](https://github.com/andreaisabelmontana/discrete-math-lab) · [prob-stats-lab](https://github.com/andreaisabelmontana/prob-stats-lab) · [business-lab](https://github.com/andreaisabelmontana/business-lab) · [research-methods-lab](https://github.com/andreaisabelmontana/research-methods-lab) · [big-history-lab](https://github.com/andreaisabelmontana/big-history-lab)
