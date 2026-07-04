# JR_PRICER

Librairie Python de pricing de produits dérivés, écrite from scratch (NumPy/SciPy) : options equity sous Black-Scholes, Heston et volatilité locale (Dupire), et produits de taux (bootstrap de courbes, swaps, caplets/floorlets, swaptions).

Le projet couvre toute la chaîne : chargement des données de marché → construction des courbes et surfaces de volatilité → calibration des modèles → pricing (analytique, Monte Carlo, Fourier) → visualisation.

## Fonctionnalités

| Module | Contenu |
|---|---|
| `pricing` | Modèles (Black-Scholes, Black-76, Heston, vol locale Dupire) et moteurs de pricing : analytique, Monte Carlo (schémas de discrétisation dédiés), Fourier (Carr-Madan). Formules Black vectorisées, calcul de vol implicite. |
| `surfaces` | Surfaces de volatilité : paramétrisations SVI / SSVI, interpolateurs 2D (bicubique, RBF), construction depuis des smiles de marché. |
| `curves` | Courbes de taux et de forward : bootstrap, courbes d'actualisation, forwards analytiques, interpolateurs 1D, smiles de vol. |
| `calibration` | Calibration de modèles : fonctions de coût, optimiseurs SciPy, régression robuste. |
| `instruments` | Instruments : options européennes/digitales/asiatiques, dépôts, swaps, jambes, dérivés de taux. |
| `market_data` | Quotes, sous-jacents, chargement de chaînes d'options. |
| `utils` | Conventions de marché : calendriers (TARGET…), day counts, fréquences, échéanciers, business day conventions. |
| `viz` | Visualisation : courbes, surfaces de vol, trajectoires Monte Carlo. |

## Installation

Nécessite Python ≥ 3.9.

```bash
git clone https://github.com/JulesRemlinger/jr-pricer.git
cd jr-pricer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Démarrage rapide

Pricing d'un call européen ATM 1 an sous Black-Scholes, en analytique et en Monte Carlo :

```python
from datetime import date

from JR_PRICER.market_data.quote import Quote
from JR_PRICER.market_data.underlying import Underlying
from JR_PRICER.curves.temporal.discount import FlatDiscountCurve
from JR_PRICER.curves.forward.analytic_forward import AnalyticForwardCurve
from JR_PRICER.surfaces.vol_surface.volsurface import FlatVol
from JR_PRICER.utils.day_count import DayCounter
from JR_PRICER.instruments.derivatives.equity.base import OptionType
from JR_PRICER.instruments.derivatives.equity.european_option import EuropeanOption
from JR_PRICER.pricing.model.blackscholes import BlackScholesModel
from JR_PRICER.pricing.engine.analytical import AnalyticalEngine
from JR_PRICER.pricing.engine.monte_carlo import MCEngine

# Marché : spot 100, taux 3 %, dividende 1 %, vol plate 20 %
ref_date = date(2026, 1, 15)
spot = Quote(100.0, "EQ")
disc = FlatDiscountCurve(0.03, DayCounter("act/365"), ref_date)
fwd = AnalyticForwardCurve(spot, disc, dividend_yield=0.01)
und = Underlying("EQ", spot, fwd, FlatVol(0.20))

# Call européen ATM 1 an
opt = EuropeanOption(und, K=100.0, start_date=ref_date,
                     maturity_date=date(2027, 1, 15), option_type=OptionType.CALL)

model = BlackScholesModel(disc)
prix_analytique = AnalyticalEngine().price([opt], model)[0]
prix_mc = MCEngine(100_000, seed=42).price([opt], model)[0]
print(f"Call ATM 1Y — analytique : {prix_analytique:.4f} | Monte Carlo : {prix_mc:.4f}")
# Call ATM 1Y — analytique : 8.8273 | Monte Carlo : 8.8050
```

## Notebooks

Les notebooks (avec leurs sorties, visibles directement sur GitHub) illustrent la librairie sur des cas complets :

| Notebook | Contenu |
|---|---|
| [DATA_01_download_aapl](notebooks/DATA_01_download_aapl.ipynb) | Téléchargement d'une chaîne d'options AAPL via yfinance (nécessite `pip install yfinance`). |
| [EQUITY_01_heston](notebooks/EQUITY_01_heston.ipynb) | Modèle de Heston : pricing Fourier, calibration sur smile, Monte Carlo. |
| [EQUITY_02_local_vol](notebooks/EQUITY_02_local_vol.ipynb) | Volatilité locale : surface de vol implicite → Dupire → pricing Monte Carlo. |
| [EQUITY_03_lv_vs_heston](notebooks/EQUITY_03_lv_vs_heston.ipynb) | Comparaison vol locale vs Heston sur données synthétiques. |
| [EQUITY_03_lv_vs_heston_aapl](notebooks/EQUITY_03_lv_vs_heston_aapl.ipynb) | Même comparaison sur données de marché AAPL réelles. |
| [RATES_01_bootstrap_curve](notebooks/RATES_01_bootstrap_curve.ipynb) | Bootstrap d'une courbe de taux depuis des instruments de marché. |
| [RATES_02_rate_options](notebooks/RATES_02_rate_options.ipynb) | Options de taux : caplets/floorlets, swaptions (Black-76). |

## Tests

La suite de tests vérifie notamment la cohérence Monte Carlo vs analytique, la parité call-put, la calibration et la reconstruction de vol locale sur surface plate :

```bash
python -m pytest
```

## Structure du projet

```
JR_PRICER/          # Package principal
├── calibration/    # Fonctions de coût, optimiseurs
├── curves/         # Courbes (discount, forward, bootstrap, interpolateurs 1D)
├── instruments/    # Options, swaps, dépôts, jambes
├── market_data/    # Quotes, sous-jacents, chargement de données
├── pricing/        # Modèles (BS, Heston, LV) et moteurs (analytique, MC, Fourier)
├── surfaces/       # Surfaces de vol (SVI, SSVI, interpolateurs 2D)
├── utils/          # Calendriers, day counts, échéanciers
└── viz/            # Visualisation
notebooks/          # Démonstrations Jupyter (equity & taux)
tests/              # Suite pytest
```

## Licence

[MIT](LICENSE) — © 2026 Jules Remlinger
