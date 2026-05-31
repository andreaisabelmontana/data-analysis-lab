# data-analysis-lab

Visual statistical inference. 12 self-contained demos, one per idea from the IE BCSAI *Fundamentals of Data Analysis* course (Devore, *Probability and Statistics for Engineering and the Sciences*).

**Live:** https://andreaisabelmontana.github.io/data-analysis-lab/

Module 1 — Sampling & estimation
1. Sampling distribution of the mean & the CLT — draw repeated samples from a chosen population and watch the sampling distribution of X̄ build up, narrow as `SE = σ/√n`, and turn Normal whatever the population shape
2. Maximum likelihood estimation — slide a parameter along the log-likelihood curve and find the MLE (exponential rate or Normal mean)
3. Method of moments vs. MLE — both estimators on the same sample, with replicate RMSE so you can see where they differ (Gamma shape) and where they coincide (exponential rate)

Module 2 — Confidence intervals
4. CI coverage — many 95% intervals, red ones miss the true mean, observed coverage approaches the stated level
5. CI width vs. n & confidence — half-width `t · s/√n` shrinks like 1/√n and grows with the confidence level

Module 3 — Hypothesis testing (one sample)
6. One-sample z / t test — statistic against a shaded rejection region, one- or two-sided
7. The p-value as a tail area — drag the statistic, the shaded tail *is* the p-value
8. Type I / II error & power — overlapping H₀ / H₁ distributions; see α, β and power = 1−β move with effect size, n and α

Module 4 — Two-sample inference
9. Two-sample t-test — pooled vs. Welch, with Satterthwaite degrees of freedom
10. Paired vs. unpaired — the same before/after data both ways; pairing removes between-subject spread and lifts the t

Module 5 — ANOVA & categorical data
11. One-way ANOVA — `F = MSB / MSE`, between-group vs. within-group variance, with df and p-value
12. Chi-square test of independence — editable contingency table, expected counts `Rᵢ·Cⱼ/N`, `χ² = Σ(O−E)²/E`

Plain HTML + canvas + KaTeX. All distribution tails (Normal, Student-t, chi-square, F) use standard numerical approximations — A&S erf, Lanczos log-gamma, and continued-fraction incomplete gamma/beta — so there are no charting or stats libraries. Indigo accent. Zero build step.

Part of the *-lab series: [discrete-math-lab](https://github.com/andreaisabelmontana/discrete-math-lab) · [prob-stats-lab](https://github.com/andreaisabelmontana/prob-stats-lab) · [business-lab](https://github.com/andreaisabelmontana/business-lab) · [research-methods-lab](https://github.com/andreaisabelmontana/research-methods-lab) · [big-history-lab](https://github.com/andreaisabelmontana/big-history-lab)
