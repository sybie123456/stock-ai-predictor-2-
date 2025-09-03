# AI Herbegin – Online Dashboard (Hersteld)

Deze versie maakt het dashboard direct werkend zonder ontbrekende code.

## Hoe starten

### Optie A – Snel starten met Streamlit
```bash
pip install -r requirements_dash.txt
streamlit run streamlit_app.py
```

### Optie B – Met omgevingsvariabelen
Je kunt `DATA_DIR` (voor lokale nieuwsbestanden) en `MODELS_DIR` (voor optionele RL modellen) instellen:
```bash
DATA_DIR=data_cache MODELS_DIR=models streamlit run streamlit_app.py
```

## Functies
- Ticker selectie + OHLC grafiek (Plotly)
- Basis technische indicatoren en datakwaliteitsoverzicht
- Optioneel lokaal nieuws met VADER-sentiment (zoek in `data_cache/`)
- Eenvoudige ARIMA-forecast (indien `statsmodels` beschikbaar is)
- CSV-export van features en voorspellingen

> Let op: RL-modellering via Stable-Baselines is optioneel en wordt alleen geprobeerd als je dit aanvinkt in de zijbalk én de lib geïnstalleerd is.
