import numpy as np


def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights) * 52))


def negative_sharpe_ratio(weights, expected_returns, rf, cov_matrix):
    return -(portfolio_return(weights, expected_returns) - rf) / portfolio_volatility(
        weights, cov_matrix
    )


def calculate_individual_beta(data, market_returns, company_name):
    company_data = data[company_name].dropna()
    aligned_market_returns = market_returns.loc[company_data.index]
    cov_with_market = np.cov(company_data, aligned_market_returns)[0, 1]
    market_variance = np.var(aligned_market_returns)
    return cov_with_market / market_variance if market_variance != 0 else np.nan


def portfolio_beta(weights, data, market_returns, data_without_ipsa):
    betas = {}
    for company in data_without_ipsa.columns:
        betas[company] = calculate_individual_beta(data, market_returns, company)
    weighted_beta = sum(
        weights[i] * betas[company]
        for i, company in enumerate(data_without_ipsa.columns)
    )
    return weighted_beta


def sortino_ratio(weights, expected_returns, data_without_ipsa, rf, target_return=0):
    portfolio_return = np.dot(weights, expected_returns)
    negative_returns = [
        min(0, r - target_return) for r in (np.dot(weights, data_without_ipsa.T) - rf)
    ]
    downside_std = np.sqrt(np.mean(np.square(negative_returns)))
    return (portfolio_return - rf) / downside_std if downside_std != 0 else np.inf


def negative_treynor_ratio(
    weights, expected_returns, rf, cov_matrix, data, market_returns, data_without_ipsa
):
    port_return = portfolio_return(weights, expected_returns)
    beta = portfolio_beta(weights, data, market_returns, data_without_ipsa)
    return -(port_return - rf) / beta if beta != 0 else np.nan


def positive_treynor_ratio(
    weights, expected_returns, rf, cov_matrix, data, market_returns, data_without_ipsa
):
    port_return = portfolio_return(weights, expected_returns)
    beta = portfolio_beta(weights, data, market_returns, data_without_ipsa)
    return (port_return - rf) / beta if beta != 0 else np.nan


def max_return_portfolio(weights, expected_returns):
    return portfolio_return(weights, expected_returns)


def negative_sortino_ratio(weights, expected_returns, data_without_ipsa, rf):
    return -sortino_ratio(weights, expected_returns, data_without_ipsa, rf)
