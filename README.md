# Alternative_Data

## Purpose

1. Given a company, do a deep dive, try to separate out strengths and weakness (determine components, risk, research helper)
2. Search for a company that looks "promising"

## Workflow

## Data sources

- [crunchbase](https://www.crunchbase.com/organization/asana/signals_and_news) (example Asana) tells us about funding rounds
- Google news: search for articles mentioning company, do senitment analysis


### Brainstorming notes

- some businesses have changed their business model during COVID, prior analysis doesn't necessarily hold up
- look for good companies in different sectors
- example of portofolio analysis with pre-covid and post-covid windows, see https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

### More detailed

1. Look at everything in S&P 500
2. Look at correlation matrix for 2019
3, Look at correlation matrix for 2020
4. Compare -- are there large shifts in the correlation between pre-covid and post-covid. Maybe for more fidelity look at Mar 2019 - Aug 2019 and Mar 2020 - current, so that we capture the period of seasonality.
5. We are really looking for large changes in correlation. Usually we would want to restrict to stocks with large trading volume, but by looking at the S&P 500 we more-or-less guarantee that anyway.
6. Try breaking the S&P into their different sectors, and look at the movement of sectors relative to one another.


Ways of looking at the correlation matrix
- use standard markowitz
- use identity
- bayesian adjusted
- black-litterman method
