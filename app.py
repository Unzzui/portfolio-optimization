import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
import time
from functions_ratios.functions import (
    portfolio_return,
    portfolio_volatility,
    negative_sharpe_ratio,
    portfolio_beta,
    negative_treynor_ratio,
    negative_sortino_ratio,
    positive_treynor_ratio,
    max_return_portfolio,
    treynor_ratio_calculate,
)

start_time = time.time()

stockList = ["^GSPC", "AAPL", "TSLA", "NVDA", "MSFT", "CAT", "V", "MA"]


# Cargamos el archivo Excel en un DataFrame de pandas
data = yf.download(stockList, period="5y")
data = data["Close"]
data = data.pct_change()
data = data.dropna()


market_returns = data["^GSPC"]
data_without_ipsa = data.drop(columns=["^GSPC"])  # Eliminamos la columna del IPSA
# Calculamos la matriz de covarianza de las rentabilidades SIN el IPSA
cov_matrix = data_without_ipsa.cov()

# Asumimos una tasa libre de riesgo de 3.57%
rf = 0.0357170688935893
# Calculamos los retornos esperados como el promedio histórico de las rentabilidades SIN el IPSA
expected_returns = data_without_ipsa.mean()
expected_returns = (1 + expected_returns) ** 252 - 1


# Restricción para que la suma de las ponderaciones sea 100%
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# Límites para las ponderaciones
bounds = tuple((0, 1) for asset in range(len(expected_returns)))

# Punto de partida igualmente ponderado
initial_guess = [1.0 / len(expected_returns) for _ in expected_returns]

# Optimización para encontrar el portafolio con el máximo Sharpe ratio
optimal_sharpe = minimize(
    negative_sharpe_ratio,
    initial_guess,
    args=(expected_returns, rf, cov_matrix),  # Agrega los argumentos adicionales aquí
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

# Optimización para encontrar el portafolio con el mínimo riesgo
optimal_variance = minimize(
    portfolio_volatility,
    initial_guess,
    args=cov_matrix,  # Agrega los argumentos adicionales aquí
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)


optimal_sortino = minimize(
    negative_sortino_ratio,
    initial_guess,
    args=(
        expected_returns,
        data_without_ipsa,
        rf,
    ),  # Agrega los argumentos adicionales aquí
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)


# Optimización para encontrar el portafolio de máximo rendimiento
optimal_return = minimize(
    lambda weights: -max_return_portfolio(weights, expected_returns),
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)


# Optimización para encontrar el portafolio con el máximo Treynor ratio
optimal_treynor = minimize(
    negative_treynor_ratio,
    initial_guess,
    args=(
        expected_returns,
        rf,
        cov_matrix,
        data,
        market_returns,
        data_without_ipsa,
    ),  # Agrega los argumentos adicionales aquí
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

# Generación de la frontera eficiente
num_portfolios = 400

efficient_frontier_data = {
    "Risk": [],
    "Return": [],
    "Sharpe Ratio": [],
    "Beta": [],
    "Treynor Ratio": [],
}

for target_return in np.linspace(
    expected_returns.min(), expected_returns.max(), num_portfolios
):
    constraints = (
        {
            "type": "eq",
            "fun": lambda x: portfolio_return(x, expected_returns) - target_return,
        },
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )

    efficient_portfolio = minimize(
        portfolio_volatility,
        initial_guess,
        args=cov_matrix,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret = portfolio_return(efficient_portfolio["x"], expected_returns)
    vol = portfolio_volatility(efficient_portfolio["x"], cov_matrix)
    sr = (ret - rf) / vol
    beta = portfolio_beta(
        efficient_portfolio["x"], data, market_returns, data_without_ipsa
    )
    treynor_ratio = (ret - rf) / beta if beta != 0 else np.nan
    efficient_frontier_data["Risk"].append(vol)
    efficient_frontier_data["Return"].append(ret)
    efficient_frontier_data["Sharpe Ratio"].append(sr)
    efficient_frontier_data["Beta"].append(beta)
    efficient_frontier_data["Treynor Ratio"].append(treynor_ratio)

# Ponderaciones iguales para cada acción
equal_weights = np.array(initial_guess)

# Calculamos las métricas para este portafolio
equal_weight_metrics = {
    "Risk": portfolio_volatility(equal_weights, cov_matrix),
    "Return": portfolio_return(equal_weights, expected_returns),
    "Sharpe Ratio": -negative_sharpe_ratio(
        equal_weights, expected_returns, rf, cov_matrix
    ),
    "Beta": portfolio_beta(equal_weights, data, market_returns, data_without_ipsa),
    "Treynor Ratio": positive_treynor_ratio(
        equal_weights,
        expected_returns,
        rf,
        cov_matrix,
        data,
        market_returns,
        data_without_ipsa,
    ),
}

# Estilo del gráfico
# Configuración del estilo del gráfico
# plt.style.use('seaborn-whitegrid')  # Estilo limpio y minimalista


# Set the figure size
plt.figure(figsize=(16, 10))

# Plot the efficient frontier
plt.scatter(
    efficient_frontier_data["Risk"],
    efficient_frontier_data["Return"],
    c=efficient_frontier_data["Sharpe Ratio"],
    cmap="viridis",
)

# Add colorbar
plt.colorbar(label="Sharpe Ratio", shrink=0.8)

# Set axis labels and formatting
plt.xlabel("Volatility (%)", fontsize=12, fontweight="bold", color="black")
plt.ylabel("Expected Return (%)", fontsize=12, fontweight="bold", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(fontsize=10, color="black")
plt.yticks(fontsize=10, color="black")

# Plot special points
for portfolio, color, label in [
    (optimal_variance, "red", "Minimum Risk"),
    (optimal_sharpe, "orange", "Maximum Return (Sharpe)"),
    (optimal_sortino, "orange", "Maximum Return (Sortino)"),
    (optimal_treynor, "blue", "Maximum Return (Treynor)"),
]:
    plt.scatter(
        portfolio_volatility(portfolio["x"], cov_matrix),
        portfolio_return(portfolio["x"], expected_returns),
        color=color,
        marker="*",
        s=150,
        label=label,
    )

# Plot equally weighted portfolio
plt.scatter(
    equal_weight_metrics["Risk"],
    equal_weight_metrics["Return"],
    color="purple",
    label="Equally Weighted",
    marker="*",
    s=150,
)

# Plot Capital Market Line (CML)
max_sharpe_return = portfolio_return(optimal_sharpe.x, expected_returns)
max_sharpe_volatility = portfolio_volatility(optimal_sharpe.x, cov_matrix)
cml_x = [0, max_sharpe_volatility]
cml_y = [rf, max_sharpe_return]
plt.plot(cml_x, cml_y, color="black", linestyle="--", linewidth=2, label="CML")

# Plot risk-free rate
plt.axhline(y=rf, color="green", linestyle=":", linewidth=2, label="Risk-Free Rate")

# Set title and legend
plt.title(
    "Efficient Frontier and Capital Market Line",
    fontsize=16,
    fontweight="bold",
    color="black",
)
plt.legend(loc="upper left", frameon=True)

# Add gridlines
plt.grid(True)

# Show the plot
plt.show()

# # Crear un DataFrame con los resultados de la optimización
# results_df = pd.DataFrame(
#     {
#         "Max Sharpe Ratio": optimal_sharpe.x,
#         "Min Risk": optimal_variance.x,
#         "Max Sortino Ratio": optimal_sortino.x,
#         "Max Return": optimal_return.x,
#         "Max Treynor Ratio": optimal_treynor.x,
#     },
#     index=data_without_ipsa.columns,
# )

# # Transponer el DataFrame para que cada fila corresponda a una cartera
# results_df = results_df.T


# Calcular el portafolio de máximo retorno (activo con mayor retorno esperado)
max_return_idx = expected_returns.argmax()
weights_max_return = np.zeros(len(expected_returns))
weights_max_return[max_return_idx] = 1


# Crear un DataFrame para mostrar los resultados
portfolios = {
    "Mínimo Riesgo": optimal_variance.x,
    "Máximo Retorno": weights_max_return,
    "Maximo Rendimiento (Sortino)": optimal_sortino.x,
    "Máximo Rendimiento (Sharpe)": optimal_sharpe.x,
}

for name, weights in portfolios.items():
    print(f"\nPortafolio: {name}")
    print("Ponderaciones de las Acciones (%):")
    for stock, weight in zip(data_without_ipsa.columns, weights * 100):
        print(f"  {stock}: {weight:.3f}%")
    print(
        f"Riesgo del Portafolio: {portfolio_volatility(weights * 100, cov_matrix):.2f}%"
    )
    print(
        f"Retorno del Portafolio: {portfolio_return(weights*100, expected_returns):.2f}%"
    )
    print(
        f"Índice de Sharpe: {-negative_sharpe_ratio(weights, expected_returns, rf, cov_matrix):.2f}"
    )
    print(
        f"Índice de Sortino: {-negative_sortino_ratio(weights, expected_returns, data_without_ipsa, rf):.2f}"
    )

    beta = portfolio_beta(weights, data, market_returns, data_without_ipsa)
    print(f"Beta del Portafolio: {beta:.2f}")
    print(
        f"Índice de Treynor: {treynor_ratio_calculate(weights, expected_returns, rf, data, market_returns, data_without_ipsa):.2f}\n"
    )

end_time = time.time()
print(f"Tiempo de ejecución: {end_time - start_time} segundos")
