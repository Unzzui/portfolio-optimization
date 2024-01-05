import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
from functions_ratios.functions import (
    portfolio_return,
    portfolio_volatility,
    negative_sharpe_ratio,
    portfolio_beta,
    negative_treynor_ratio,
    negative_sortino_ratio,
    positive_treynor_ratio,
    max_return_portfolio,
)

stockList = ["^GSPC", "AAPL", "TSLA", "NVDA"]


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
# Optimización para encontrar el portafolio de máximo rendimiento
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

# Tabla de resultados
results_df = pd.DataFrame(efficient_frontier_data)
results_df.index += 1  # Para que los índices comiencen en 1
results_df.index.name = "Portafolio de Markowitz"

# Calculamos el portafolio con el mayor ratio de Sharpe en la frontera eficiente
max_sharpe_idx = np.argmax(efficient_frontier_data["Sharpe Ratio"])
max_sharpe_portfolio = {
    "Risk": efficient_frontier_data["Risk"][max_sharpe_idx],
    "Return": efficient_frontier_data["Return"][max_sharpe_idx],
    "Sharpe Ratio": efficient_frontier_data["Sharpe Ratio"][max_sharpe_idx],
    "Beta": efficient_frontier_data["Beta"][max_sharpe_idx],
    "Treynor Ratio": efficient_frontier_data["Treynor Ratio"][max_sharpe_idx],
}

# Calculamos el portafolio con el menor riesgo en la frontera eficiente
min_risk_idx = np.argmin(efficient_frontier_data["Risk"])
min_risk_portfolio = {
    "Risk": efficient_frontier_data["Risk"][min_risk_idx],
    "Return": efficient_frontier_data["Return"][min_risk_idx],
    "Sharpe Ratio": efficient_frontier_data["Sharpe Ratio"][min_risk_idx],
    "Beta": efficient_frontier_data["Beta"][min_risk_idx],
    "Treynor Ratio": efficient_frontier_data["Treynor Ratio"][min_risk_idx],
}

max_treynor_idx = np.argmax(efficient_frontier_data["Treynor Ratio"])
max_treynor_portfolio = {
    "Risk": efficient_frontier_data["Risk"][max_treynor_idx],
    "Return": efficient_frontier_data["Return"][max_treynor_idx],
    "Sharpe Ratio": efficient_frontier_data["Sharpe Ratio"][max_treynor_idx],
    "Beta": efficient_frontier_data["Beta"][max_treynor_idx],
    "Treynor Ratio": efficient_frontier_data["Treynor Ratio"][max_treynor_idx],
}
# Ponderaciones iguales para cada acción
equal_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))

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


equal_weight_df = pd.DataFrame(
    [equal_weight_metrics], index=["Portafolio Igualmente Ponderado"]
)

max_sharpe_df = pd.DataFrame(
    [max_sharpe_portfolio], index=["Portafolio de Máximo Sharpe"]
)
min_risk_df = pd.DataFrame([min_risk_portfolio], index=["Portafolio de Mínimo Riesgo"])
max_treynor_df = pd.DataFrame(
    [max_treynor_portfolio], index=["Portafolio de Máximo Treynor"]
)
# Concatenamos los DataFrames
results_with_special_portfolios = pd.concat(
    [results_df, max_sharpe_df, min_risk_df, max_treynor_df, equal_weight_df]
)


# Estilo del gráfico
# Configuración del estilo del gráfico
# plt.style.use('seaborn-whitegrid')  # Estilo limpio y minimalista
plt.figure(figsize=(16, 10), dpi=400)

# Frontera eficiente
plt.scatter(
    efficient_frontier_data["Risk"],
    efficient_frontier_data["Return"],
    c=efficient_frontier_data["Sharpe Ratio"],
    cmap="viridis",
)

plt.colorbar(label="Índice de Sharpe", shrink=0.8)
# Formato de ejes y etiquetas
plt.xlabel("Volatilidad (%)", fontsize=12, fontweight="bold", color="black")
plt.ylabel("Rendimiento Esperado (%)", fontsize=12, fontweight="bold", color="black")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(fontsize=10, color="black")
plt.yticks(fontsize=10, color="black")
# Asumiendo que optimal_sharpe es el resultado de la optimización para el máximo Sharpe ratio
max_sharpe_return = portfolio_return(optimal_sharpe.x, expected_returns)
max_sharpe_volatility = portfolio_volatility(optimal_sharpe.x, cov_matrix)
# Puntos especiales
for portfolio, color, label in [
    (optimal_variance, "red", "Mínimo Riesgo"),
    (optimal_sharpe, "orange", "Máximo Rendimiento (Sharpe)"),
    (optimal_sortino, "orange", "Máximo Rendimiento (Sortino)"),
    (optimal_treynor, "blue", "Máximo Rendimiento (Treynor)"),
]:
    plt.scatter(
        portfolio_volatility(portfolio["x"], cov_matrix),
        portfolio_return(portfolio["x"], expected_returns),
        color=color,
        marker="*",
        s=150,
        label=label,
    )

plt.scatter(
    equal_weight_metrics["Risk"],
    equal_weight_metrics["Return"],
    color="purple",
    label="Igualmente Ponderado",
    marker="*",
    s=150,
)

# Línea del mercado de capitales (CML)
# cml_x = [0, max(efficient_frontier_data["Risk"])]
# cml_y = [rf, optimal_variance.fun]

cml_x = [0, max_sharpe_volatility]
cml_y = [rf, max_sharpe_return]
plt.plot(cml_x, cml_y, color="black", linestyle="--", linewidth=2, label="CML")

# Línea horizontal para la tasa libre de riesgo
plt.axhline(
    y=rf, color="green", linestyle=":", linewidth=2, label="Tasa Libre de Riesgo"
)

# Título y leyenda fuera del gráfico
plt.title(
    "Frontera Eficiente de Markowitz y Línea CML",
    fontsize=16,
    fontweight="bold",
    color="black",
)
plt.legend(loc="upper left", frameon=True)

# Agregar líneas de cuadrícula
plt.grid(True)

plt.show()


# Calcular el portafolio de máximo retorno (activo con mayor retorno esperado)
max_return_idx = expected_returns.argmax()
weights_max_return = np.zeros(len(expected_returns))
weights_max_return[max_return_idx] = 1


# Función para calcular el ratio de Treynor de un portafolio
def treynor_ratio(
    weights, expected_returns, rf, data, market_returns, data_without_ipsa
):
    port_return = portfolio_return(weights, expected_returns)
    beta = portfolio_beta(weights, data, market_returns, data_without_ipsa)
    return (port_return - rf) / beta if beta != 0 else np.nan


# Crear un DataFrame para mostrar los resultados
portfolios = {
    "Mínimo Riesgo": optimal_variance.x,
    "Máximo Retorno": weights_max_return,
    "Maximo Rendimiento (Sortino)": optimal_sortino.x,
    "Máximo Rendimiento (Sharpe)": optimal_sharpe.x,
    "Máximo Rendimiento (Treynor)": optimal_treynor.x,
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
        f"Índice de Treynor: {treynor_ratio(weights, expected_returns, rf, data, market_returns, data_without_ipsa):.2f}\n"
    )
