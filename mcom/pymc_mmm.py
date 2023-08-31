from datetime import datetime, timedelta
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pymc as pm
from pytensor import tensor as pt

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100


# DATA IMPORT

data_df = pd.read_csv()
# placeholders for variables recieved from data pipeline
target_col = "target"
ad_input_column = "ad_spend"
# from data pipeline we a recieving spend per column + seasonal variables from prophet


data_df_original = data_df.copy()
data_df_original.reset_index(inplace=True, drop=True)
spend_df = data_df_original.sum(axis=0).reset_index()
spend_df.rename(columns={"index": "input_variable", 0: ad_input_column}, inplace=True)
spend_df[target_col] = spend_df.loc[spend_df["input_variable"] == target_col][
    ad_input_column
].values[0]
spend_df.drop(spend_df.index[[0, 1]], inplace=True)

data_df.rename(
    columns={
        "seasonality_daily": "seasonality_1",
        "seasonality_weekly": "seasonality_7",
        "seasonality_7w": "seasonality_52",
        "seasonality_yearly": "seasonality_365",
    },
    inplace=True,
)

# creatiing list of regressors
columns_df = list(data_df.keys())
list_exclude = [target_col, "date"]
regressors = list(filter(lambda x: x not in list_exclude, columns_df))


# finalising dataprep
data_df.reset_index(inplace=True)
data_df.drop(columns=["date"], inplace=True)


# dividing features lists
seasonality_list = [
    "trend",
    "seasonality_1",
    "seasonality_7",
    "seasonality_52",
    "seasonality_365",
    "seasonality_pay_cycle",
    "holidays",
]

media_list = list(filter(lambda x: x not in seasonality_list, regressors))
saturation_list = ["paid_social"]

scaler = MinMaxScaler()
data_df.loc[:, regressors] = scaler.fit_transform(data_df.loc[:, regressors])


divider = 100

data_df.loc[:, target_col] = (
    data_df.loc[:, target_col].values.reshape(-1, 1)
) / divider


# PRIORS PREPARATION

# preparing priors or expert opinion on possible Cost per conversion per channel to set up prior distribution
priors_df = pd.read_csv()


sources = []
for index, row in priors_df.iterrows():
    # Split the column name by underscores
    part = row.channel.split("_")[2]
    sources.append(part)
sources = list(set(sources))

mass_dct = {
    "mobile_display": 0.9,
    "mailing": 0.8,
    "display": 0.6,
    "paid_search": 0.9,
    "paid_social": 0.6,
}

for index, row in priors_df.iterrows():
    for key, value in mass_dct.items():
        if key in row["channel"]:
            priors_df.loc[index, "mass"] = value


converged_priors = pd.DataFrame()
hyperparams_dict = {}
hyperparams_dict_distr = {}


for channel in priors_df[~priors_df["channel"].isin(hyperparams_dict.keys())].channel:
    lower_value = priors_df[priors_df["channel"] == channel]["lower_bound"].values[0]
    upper_value = priors_df[priors_df["channel"] == channel]["upper_bound"].values[0]
    mass = priors_df[priors_df["channel"] == channel]["mass"].values[0]
    if ("mobile_display") in channel:
        MASS = mass
        init_guess = dict(mu=2, sigma=10)
        mass_below_lower = 0
        options = {"maxiter": 100000}
        distribution = pm.Gamma
    if ("mailing") in channel:
        MASS = mass
        init_guess = dict(mu=2, sigma=10)
        mass_below_lower = 0
        options = {"maxiter": 100000}
        distribution = pm.Gamma
    if ("display") in channel:
        MASS = mass
        init_guess = dict(mu=2, sigma=10)
        mass_below_lower = 0
        options = {"maxiter": 100000}
        distribution = pm.Gamma
    if ("paid_search") in channel:
        MASS = mass
        init_guess = dict(mu=2, sigma=10)
        mass_below_lower = 0
        options = {"maxiter": 100000}
        distribution = pm.Gamma
    if ("paid_social") in channel:
        MASS = mass
        init_guess = dict(mu=2, sigma=10)
        mass_below_lower = 0
        options = {"maxiter": 100000}
        distribution = pm.Gamma
    try:
        hyperparams_dict[f"{channel}"] = pm.find_constrained_prior(
            distribution,
            lower=lower_value,
            upper=upper_value,
            mass=mass,
            init_guess=init_guess,
            mass_below_lower=mass_below_lower,
        )
        hyperparams_dict_distr[f"{channel}"] = [distribution]

        new_row = pd.Series(
            {
                "channel": channel,
                "distribution": str(hyperparams_dict_distr[f"{channel}"][0]),
                "hyperparams": hyperparams_dict[f"{channel}"],
            },
        )
        converged_priors = pd.concat(
            [converged_priors, pd.DataFrame([new_row], columns=new_row.index)]
        ).reset_index(drop=True)
    except:
        print(f"{channel} failed")
        pass


def sat_function(x_t, mu, scale=False):
    y = (1 - np.exp(-mu * x_t)) / (1 + np.exp(-mu * x_t))

    if scale:
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

    return y


def weibull_adstock_at(
    x_t, alpha: float, theta: int = 0, L=12, normalize=True, eval=False
):
    w = pt.power(x=alpha, y=pt.power(((pt.ones(L).cumsum() - 1) - theta), 2))
    # w = (alpha**((pt.ones(L).cumsum()-1)-theta)**2)

    xx = pt.stack(
        [pt.concatenate([pt.zeros(i), x_t[: x_t.shape[0] - i]]) for i in range(L)]
    )

    if not normalize:
        y = pt.dot(w, xx)
    else:
        y = pt.dot(w / pt.sum(w), xx)

    if eval:
        return y.eval()
    else:
        return y


def beta_hill_saturation_build(
    x: np.array, alpha: float, gamma: float, beta: float, scale=False
) -> np.array:
    gammatrans = (pt.max(x) - pt.min(x)) * gamma

    if scale == False:
        sat_curve = beta - (pt.power(x=x, y=alpha) * beta) / (
            pt.power(x=x, y=alpha) + pt.power(x=gammatrans, y=alpha)
        )
    else:
        sat_curve = beta - (x**alpha * beta) / (x**alpha + gammatrans**alpha)
        sat_curve = sat_curve * (pt.max(x) - pt.min(x)) + pt.min(x)
    return sat_curve


print("setting up model")

# model specification
seasonality_value_mean = []
ad_spend_variable = []
media_coef = []
media_conversions = []
media_coef_value_uplift = []

media_list_dict = {}
media_list_data_dict = {}
saturation_dict = {}

date_day = data_df.date.to_numpy()
coords = {"date_day": date_day}


intercept_mu = 2
intercept_sigma = 2.5
sigma_sigma = 3
seasonality_trend_mu = 0
seasonality_trend_sigma = 1


with pm.Model(coords=coords) as base_model:
    # --- data containers ---
    ## list of media variables spend

    for regressor in media_list:
        media_list_data_dict[regressor] = pm.MutableData(
            name=f"{regressor}_data",
            value=data_df[regressor].values,
        )

    ## control variables
    seasonality_trend_data = pm.MutableData(
        name="seasonality_trend_data",
        value=(
            data_df["seasonality_365"] + data_df["seasonality_52"] + data_df["trend"]
        ).values,
    )

    # --- priors ---
    ## intercept
    intercept_coef = pm.Normal("intercept_coef", mu=intercept_mu, sigma=intercept_sigma)

    ## standard deviation of the normal likelihood
    sigma = pm.HalfNormal("sigma", sigma_sigma)

    seasonality_trend_coef = pm.Normal(
        name="seasonality_trend_coef",
        mu=seasonality_trend_mu,
        sigma=seasonality_trend_sigma,
    )

    ads_alpha = pm.Beta(f"{regressor}_ads_alpha", alpha=3, beta=3)
    ads_theta = pm.Gamma(f"{regressor}_ads_theta", mu=2, sigma=3)
    sat_gamma = pm.Beta(f"{regressor}_sat_gamma", alpha=3, beta=3)
    sat_alpha = pm.Gamma(f"{regressor}_sat_alpha", mu=3, sigma=3)

    ## media variables coefs
    for regressor in media_list:
        if regressor in hyperparams_dict.keys():
            media_list_dict[f"{regressor}"] = hyperparams_dict_distr[regressor][0](
                f"{regressor}_coef", **hyperparams_dict[regressor]
            )
        else:
            media_distr_sigma = 0.8
            media_list_dict[regressor] = pm.HalfNormal(
                f"{regressor}_coef", sigma=media_distr_sigma
            )
        saturation_dict[f"{regressor}"] = pm.Gamma(
            f"{regressor}_half_sat", alpha=3, beta=1
        )

    ## WIP Time-Varying coefficients
    # slopes = pm.GaussianRandomWalk(
    #     name="slopes",
    #     sigma=sigma_slope,
    #     init_dist=pm.distributions.continuous.Normal.dist(
    #         name="init_dist", mu=0, sigma=2
    #     ),
    #     dims="date",
    # )
    # adstock = pm.Deterministic(
    #     name="ads_adstock", var=geometric_adstock(x=z_scaled_, alpha=alpha, l_max=12), dims="date"
    # )
    # adstock_saturated = pm.Deterministic(
    #     name="z_adstock_saturated",
    #     var=logistic_saturation(x=z_adstock, lam=lam),
    #     dims="date",
    # )
    # Ads_effect = pm.Deterministic(
    #     name="z_effect", var=pm.math.exp(slopes) * z_adstock_saturated, dims="date"
    # )

    # --- model parametrisation ---

    for regressor in media_list:
        # saturation only channels
        if regressor in saturation_list:
            ad_spend = media_list_data_dict[regressor]
            channel_beta = media_list_dict[regressor]
            regressor_estimated_value = beta_hill_saturation_build(
                ad_spend, alpha=sat_alpha, gamma=sat_gamma, beta=channel_beta
            )
            media_coef.append(regressor_estimated_value)

        else:
            # saturation + adstock decay channels
            ad_spend = media_list_data_dict[regressor]
            channel_beta = media_list_dict[regressor]
            adstock_spend = weibull_adstock_at(
                ad_spend, alpha=ads_alpha, theta=ads_theta
            )
            regressor_estimated_value = beta_hill_saturation_build(
                adstock_spend, alpha=sat_alpha, gamma=sat_gamma, beta=channel_beta
            )
            media_coef.append(regressor_estimated_value)

    media_contribution = sum(media_coef)

    seasonality_trend_contribution = pm.math.dot(
        seasonality_trend_coef, seasonality_trend_data
    )

    mu = intercept_coef + seasonality_trend_contribution + media_contribution

    # --- likelihood ---
    pm.Normal(
        name=target_col,
        sigma=sigma,
        mu=mu,
        observed=data_df[target_col].values,
    )

    # --- prior samples ---
    base_model_prior_predictive = pm.sample_prior_predictive()

# visualising causal graph
pm.model_to_graphviz(model=base_model)

# sample model

draws = 2000
chains = 4
tune = 1000

with base_model:
    base_model_trace = pm.sampling_jax.sample_numpyro_nuts(
        tune=tune, target_accept=1, draws=draws, chains=chains, random_seed=10
    )
    base_model_posterior_predictive = pm.sample_posterior_predictive(
        trace=base_model_trace
    )


# exploring unscaled fit
posterior_predictive_likelihood = base_model_posterior_predictive.posterior_predictive[
    target_col
].stack(sample=("chain", "draw"))
posterior_predictive_likelihood_inv = (posterior_predictive_likelihood) * divider
y_true = np.concatenate(
    ((data_df[target_col]).values.reshape(-1, 1)) * divider,
    axis=0,
)

y_pred = np.round(posterior_predictive_likelihood_inv.mean(axis=1), 0)


# add MAE
RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))
MAPE = np.mean(np.abs((y_true - y_pred) / y_true))
SMAPE = np.mean(abs(y_pred - y_true) / ((abs(y_true) + abs(y_pred)) / 2))
print(f"RMSE: {RMSE}")
print(f"MAPE: {MAPE}")
print(f"SMAPE: {SMAPE}")


"""
import seaborn as sns
fig, ax = plt.subplots()
sns.lineplot(
    x=date_geo,
    y=y_true,
    color="black",
    label="y_scaled (scaled)",
    #ax=ax,
)

sns.lineplot(
    x=date_geo,
    y=y_pred,
    color="C2",
    label="posterior predictive mean",
    #ax=ax,
)
ax.legend(loc="upper left")
ax.set(title="Base Model - Posterior Predictive Samples")
"""


# az.plot_ppc(base_model_posterior_predictive, var_names=[target_col])


summary_df = az.summary(
    base_model_trace,
    var_names=[
        "ad_spend",
        "impressions",
        "Other_campaigns_Othercampaigns_1111",
        "seasonality",
        "trend",
        # "sigma",
        "intercept",
        "holidays",
    ],
    filter_vars="like",
)
