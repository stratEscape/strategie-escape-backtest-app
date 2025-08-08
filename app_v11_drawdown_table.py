
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(page_title="Stratégie Escape - Max Relative DD (Equity) — Table Mois x Année", layout="wide")

# --- Helpers existants ---
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
        pd.DataFrame({"Datetime": df2["Datetime"], "Balance": balance, "Equity": equity})
        .dropna(subset=["Datetime","Equity"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    return out

def monthly_relative_dd(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["EquityPeak"] = tmp["Equity"].cummax()
    tmp["Rel_Drawdown_%"] = (tmp["Equity"] - tmp["EquityPeak"]) / tmp["EquityPeak"] * 100.0
    tmp["Year"] = tmp["Datetime"].dt.year
    tmp["Month"] = tmp["Datetime"].dt.month
    month_map = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Juin",7:"Juil",8:"Août",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
    tmp["MonthName"] = tmp["Month"].map(month_map)

    # Max Rel DD par mois (Equity) -> on prend la valeur minimale (la plus négative) dans le mois
    m = tmp.groupby(["Year","Month","MonthName"])["Rel_Drawdown_%"].min().reset_index()

    # Pivot Mois (lignes) x Année (colonnes)
    pivot = m.pivot_table(index=["Month","MonthName"], columns="Year", values="Rel_Drawdown_%", aggfunc="min").sort_index()
    # Réordonner les lignes par n° de mois (1->12) et ne garder que MonthName à l'affichage
    pivot = pivot.reset_index().sort_values("Month").set_index("MonthName").drop(columns=["Month"])

    return pivot

def style_dd_table(pivot: pd.DataFrame) -> pd.io.formats.style.Styler:
    # Trouver la valeur minimale (la plus négative) globale
    min_val = np.nanmin(pivot.values)
    # Palette douce bleu -> blanc pour le reste (valeurs en % négatives)
    # On normalise sur l'amplitude des valeurs pour un dégradé cohérent
    vmax = 0.0
    vmin = np.nanmin(pivot.values) if not np.isnan(min_val) else -1.0

    def highlight_max(s):
        return [("font-weight:700; color:white; background-color:#B00020") if v == min_val else "" for v in s]

    styled = (
        pivot.style
        .format("{:.2f}%")
        .background_gradient(axis=None, cmap="Blues", vmin=vmin, vmax=vmax)
        .apply(highlight_max, axis=1)
    )
    return styled

st.title("Max Relative Drawdown (Equity) — Tableau Mois x Année")
st.caption("Affiche le **max drawdown relatif** sur l'Equity pour chaque mois, en lignes, et les années en colonnes. La valeur globale la plus faible est **mise en rouge**.")

uploaded = st.file_uploader("Dépose ton CSV MT5 (Date/Heure/Balance/Equity)", type=["csv"])

if uploaded is not None:
    try:
        df_raw = robust_read_csv(uploaded)
        uploaded.seek(0)
        df = normalize_columns(df_raw)

        if df.empty:
            st.error("Aucune donnée exploitable. Vérifie les colonnes Date/Heure/Balance/Equity.")
        else:
            pivot = monthly_relative_dd(df)

            st.subheader("Tableau Mois (lignes) x Années (colonnes) — Rel DD %")
            styled = style_dd_table(pivot)
            st.dataframe(styled, use_container_width=True)

            # Exports
            col1, col2 = st.columns(2)
            csv_bytes = pivot.to_csv().encode("utf-8")
            col1.download_button("⬇️ Export CSV", data=csv_bytes, file_name="monthly_rel_dd_table.csv", mime="text/csv")

            # Petit récap global
            min_val = np.nanmin(pivot.values)
            # Trouver coordonnées de la valeur min
            pos = np.where(pivot.values == min_val)
            if len(pos[0])>0:
                r, c = pos[0][0], pos[1][0]
                month = pivot.index[r]
                year = pivot.columns[c]
                st.info(f"📉 **Min global**: {min_val:.3f}% — **{month} {year}**")
    except Exception as e:
        st.exception(e)
else:
    st.info("Charge un CSV pour générer le tableau.")
