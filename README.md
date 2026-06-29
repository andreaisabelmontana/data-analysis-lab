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

## Run & test

Open `index.html` in any browser, or serve the folder statically — no dependencies, no build.

Tests use Node 24+ with the built-in runner only (`node:test` + `node:assert`, no npm install):

```
node --test
```

41 tests cover the cores listed above (summary stats, correlation/regression, transforms, group-by, distribution tails, the seeded RNG, and every inference routine).

## Coursework

Hands-on apps I built for the course, both live in the browser:
[VigiView](https://andreaisabelmontana.github.io/vigiview/) (adverse drug-event explorer) ·
[ShopSmart](https://andreaisabelmontana.github.io/shopsmart/) (cheapest-basket price comparison).

Part of the *-lab series: [discrete-math-lab](https://github.com/andreaisabelmontana/discrete-math-lab) · [prob-stats-lab](https://github.com/andreaisabelmontana/prob-stats-lab) · [business-lab](https://github.com/andreaisabelmontana/business-lab) · [research-methods-lab](https://github.com/andreaisabelmontana/research-methods-lab) · [big-history-lab](https://github.com/andreaisabelmontana/big-history-lab)
