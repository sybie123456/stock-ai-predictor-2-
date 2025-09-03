# AI Aandelen Adviseur — Online Leren (Continual Learning)

Deze versie breidt je originele project uit met **online leren**, **anti-vergeet-mechanismen** en **extra datasources** (koers + nieuws).
Het doel: doorlopend bijleren op nieuwe data, zonder te vergeten wat eerder geleerd is.

## Belangrijkste wijzigingen
- **Online learning loop** (`src/online_trainer.py` + `main.py`): haalt periodiek nieuwe data op, traint kort verder, slaat checkpoints op.
- **Anti-forgetting**
  - **Mixed replay**: de omgeving sampled zowel uit **historische** als **nieuwe** tijdvensters (uit een lokale dataset-cache).
  - **Validation guard**: vóór het overschrijven van het “beste” model wordt performance gecheckt op een vaste historische **validatieset**. Bij achteruitgang wordt **niet** opgeslagen (en optioneel rollback).
- **Meer data & nieuws**
  - **Koersen**: `yfinance` + optionele **Finnhub** (als API key gezet) als fallback/extra velden.
  - **Nieuws**: **NewsAPI.org** + optionele **Finnhub company news**. Teksten worden geanalyseerd met Hugging Face sentiment pipeline.
- **Robuuste feature set**: prijsfeatures + technische indicatoren + geaggregeerd sentimentsignaal per dag.

## Snel starten
1. Installeer requirements: `pip install -r requirements.txt`
2. Zet je API keys in `config.py` (optioneel maar aanbevolen voor extra nieuws/koersen).
3. Run éénmalig om een model te trainen:
   ```bash
   python main.py --ticker MSFT --train-initial 200000
   ```
4. Start de **online loop** (bijv. elk uur 5k stappen bijleren):
   ```bash
   python main.py --ticker MSFT --online --online-steps 5000 --interval 60
   ```

## Bestanden
- `main.py` — CLI voor initiale training en/of online loop.
- `src/online_trainer.py` — continue bijleerlogica + validatie + checkpointing.
- `src/trading_env.py` — omgeving die mixt tussen historische en recente vensters (mixed replay).
- `src/data_collector.py` — yfinance + optionele Finnhub.
- `src/news_providers.py` — NewsAPI.org en Finnhub nieuws.
- `src/news_analyzer.py` — sentiment (Hugging Face pipeline). 
- `src/feature_engineer.py` — technische indicatoren en feature-samenvoeging.
- `src/utils.py` — I/O, performance-metrics, dataset-cache.

## Notities
- **Geen catastrofale vergeetachtigheid** is nooit 100% te garanderen, maar met mixed replay + validatie-guard en kleine “online” leersessies beperk je het risico sterk.
- Dit is SB3 **PPO**-gebaseerd en laat **doorgaand trainen** toe met hergebruik van checkpoints.
