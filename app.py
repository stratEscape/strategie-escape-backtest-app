
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(page_title="Stratégie Escape - Backtest MT5 Analyzer", layout="wide")

# --- Palette (finance, bleus) ---
COLOR_BG = "#0D1117"
COLOR_EQUITY = "#1E3A5F"
COLOR_BALANCE = "#4B9CD3"
COLOR_DD = "#A0C4FF"
COLOR_TEXT = "#FFFFFF"

# --- Helpers ---
def robust_read_csv(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df_try = pd.read_csv(file, encoding=enc, sep=sep, engine="python")
                if df_try.shape[1] >= 2:
                    return df_try
            except Exception as e:
                last_err = e
                try:
                    file.seek(0)
                except Exception:
                    pass
    raise RuntimeError(f"Echec lecture CSV. Dernière erreur: {last_err}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    lower_cols = {c: c.lower() for c in df2.columns}
    df2.rename(columns=lower_cols, inplace=True)
    # Construire Datetime
    if "date" in df2.columns and "heure" in df2.columns:
        dt = pd.to_datetime(df2["date"].astype(str) + " " + df2["heure"].astype(str), errors="coerce", dayfirst=True)
    elif "date" in df2.columns and "time" in df2.columns:
        dt = pd.to_datetime(df2["date"].astype(str) + " " + df2["time"].astype(str), errors="coerce", dayfirst=True)
    elif "datetime" in df2.columns:
        dt = pd.to_datetime(df2["datetime"], errors="coerce", dayfirst=True)
    elif "time" in df2.columns:
        dt = pd.to_datetime(df2["time"], errors="coerce", dayfirst=True)
    else:
        dt = pd.to_datetime(df2.iloc[:,0], errors="coerce", dayfirst=True)
    df2["Datetime"] = dt

    # détecter colonnes
    balance_col, equity_col = None, None
    for c in df2.columns:
        lc = c.lower()
        if balance_col is None and "bal" in lc:
            balance_col = c
        if equity_col is None and ("equit" in lc or "équ" in lc):
            equity_col = c

    def to_float(s):
        if s is None or s not in df2.columns: 
            return None
        return pd.to_numeric(
                df2[s].astype(str)
                    .str.replace("\u00A0", "", regex=False)
                    .str.replace("\u202f", "", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False),
                errors="coerce"
            )

    balance = to_float(balance_col) if balance_col else None
    equity = to_float(equity_col) if equity_col else None

    out = (
    pd.DataFrame({
        "Datetime": df2["Datetime"],
        "Balance": balance,
        "Equity": equity
    })
    .dropna(subset=["Datetime", "Equity"])
    .sort_values("Datetime")
    .reset_index(drop=True)
)
    return out

def compute_metrics(df: pd.DataFrame) -> dict:
    res = {}
    if df.empty:
        return res
    start_val = float(df["Equity"].iloc[0])
    end_val = float(df["Equity"].iloc[-1])
    res["Net P&L ($)"] = end_val - start_val
    res["Return (%)"] = (end_val - start_val) / start_val * 100.0

    # Equity drawdowns
    df["EquityPeak"] = df["Equity"].cummax()
    df["Rel_Drawdown_%"] = (df["Equity"] - df["EquityPeak"]) / df["EquityPeak"] * 100.0
    df["Rel_Drawdown_$"] = (df["Equity"] - df["EquityPeak"])

    # Global max DD (%)
    idx = df["Rel_Drawdown_%"].idxmin()
    if pd.notna(idx):
        res["Max Rel DD (%)"] = float(df.loc[idx, "Rel_Drawdown_%"])
        res["Max Rel DD ($)"] = float(df.loc[idx, "Rel_Drawdown_$"])
        res["Date Max Rel DD"] = df.loc[idx, "Datetime"]
    else:
        res["Max Rel DD (%)"] = np.nan
        res["Max Rel DD ($)"] = np.nan
        res["Date Max Rel DD"] = None

    # Balance drawdown (optionnel si balance dispo)
    if not df["Balance"].isna().all():
        df["BalPeak"] = df["Balance"].cummax()
        df["Bal_Rel_DD_%"] = (df["Balance"] - df["BalPeak"]) / df["BalPeak"] * 100.0
        idb = df["Bal_Rel_DD_%"].idxmin()
        if pd.notna(idb):
            res["Max Rel DD Balance (%)"] = float(df.loc[idb, "Bal_Rel_DD_%"])
    return res

def monthly_relative_dd(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["EquityPeak"] = tmp["Equity"].cummax()
    tmp["Rel_Drawdown_%"] = (tmp["Equity"] - tmp["EquityPeak"]) / tmp["EquityPeak"] * 100.0
    tmp["Rel_Drawdown_$"] = (tmp["Equity"] - tmp["EquityPeak"])
    tmp["YearMonth"] = tmp["Datetime"].dt.to_period("M")

    def month_dd_stats(g):
        idx = g["Rel_Drawdown_%"].idxmin()
        return pd.Series({
            "Rel_DD_max_%": g.loc[idx, "Rel_Drawdown_%"],
            "Rel_DD_max_$": g.loc[idx, "Rel_Drawdown_$"],
            "Date_DD_max": g.loc[idx, "Datetime"],
            "Equity_Peak": g.loc[idx, "EquityPeak"],
            "Equity_Low": g.loc[idx, "Equity"],
        })

    m = tmp.groupby("YearMonth").apply(month_dd_stats).reset_index()
    return m

def equity_chart(df: pd.DataFrame):
    df2 = df.copy()
    df2["EquityPeak"] = df2["Equity"].cummax()
    df2["Rel_Drawdown_%"] = (df2["Equity"] - df2["EquityPeak"]) / df2["EquityPeak"] * 100.0

    x_num = mdates.date2num(df2["Datetime"].dt.to_pydatetime())

    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.plot(df2["Datetime"], df2["Equity"], linewidth=2, label="Equity", color=COLOR_EQUITY)
    if not df2["Balance"].isna().all():
        ax.plot(df2["Datetime"], df2["Balance"], linestyle="--", linewidth=1.5, label="Balance", color=COLOR_BALANCE)

    ax2 = ax.twinx()
    ax2.fill_between(x_num, df2["Rel_Drawdown_%"].values, 0, alpha=0.3, label="Drawdown (%)", color=COLOR_DD)

    ax.set_title("Courbe Equity / Balance & Drawdown (%)", color=COLOR_TEXT)
    ax.set_xlabel("Temps", color=COLOR_TEXT)
    ax.set_ylabel("USD", color=COLOR_TEXT)
    ax2.set_ylabel("Drawdown (%)", color=COLOR_TEXT)

    ax.tick_params(axis='x', colors=COLOR_TEXT)
    ax.tick_params(axis='y', colors=COLOR_TEXT)
    ax2.tick_params(axis='y', colors=COLOR_TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    fig.tight_layout()
    return fig

st.title("Stratégie Escape — Analyse Backtest MT5 (V1)")
st.caption("Upload un CSV (Date/Heure/Balance/Equity). Affiche le max Relative Drawdown sur l'Equity (global + par mois), et la courbe Equity/Balance.")

uploaded = st.file_uploader("Dépose ton fichier CSV MT5", type=["csv"])

if uploaded is not None:
    try:
        df_raw = robust_read_csv(uploaded)
        uploaded.seek(0)
        df = normalize_columns(df_raw)

        if df.empty:
            st.error("Aucune donnée exploitable. Vérifie les colonnes Date/Heure/Balance/Equity.")
        else:
            # --- Résumé ---
            st.subheader("Résumé")
            metrics = compute_metrics(df)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Net P&L ($)", f"{metrics.get('Net P&L ($)', float('nan')):,.2f}")
            mcol2.metric("Return (%)", f"{metrics.get('Return (%)', float('nan')):,.2f}")
            mcol3.metric("Max Rel DD (%)", f"{metrics.get('Max Rel DD (%)', float('nan')):,.3f}")
            date_dd = metrics.get("Date Max Rel DD", None)
            mcol4.metric("Date Max Rel DD", date_dd.strftime('%Y-%m-%d') if date_dd else "-")

            # --- Tableau mensuel ---
            st.subheader("Max Relative Drawdown (Equity) — par mois")
            monthly = monthly_relative_dd(df)
            st.dataframe(monthly, use_container_width=True)

            # Export CSV
            csv_bytes = monthly.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger le tableau (CSV)", csv_bytes, file_name="monthly_relative_dd_equity.csv", mime="text/csv")

            # --- Graphique ---
            st.subheader("Courbe Equity / Balance & Drawdown")
            fig = equity_chart(df)
            st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.exception(e)
else:
    st.info("Charge un CSV pour démarrer. Astuce: exports MT5 peuvent être encodés en UTF‑16 ou CP1252; le lecteur est robuste.")
