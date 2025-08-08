# strategie-escape-backtest-app

**Stratégie Escape — Backtest MT5 Analyzer (V1)**

## Déploiement sur Streamlit Community Cloud
1) Crée un repo GitHub et ajoute ces fichiers à la racine :
   - `app.py`
   - `requirements.txt`
2) Va sur `share.streamlit.io` → **Deploy an app**.
3) Sélectionne ton repo, branche principale, et laisse **App file** = `app.py`.
4) Déploie → URL publique immédiate.

## Déploiement local
```bash
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

## Notes
- Encodages CSV gérés: UTF‑16 / CP1252 / UTF‑8 (auto).
- Colonnes attendues: Date/Heure/Balance/Equity (noms tolérants).
