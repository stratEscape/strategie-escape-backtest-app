
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Stratégie Escape - Max Relative DD (Equity) — Table Mois x Année", layout="wide")

# --- Robust CSV reader ---
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

# --- Normalize columns & build Datetime ---
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    lower_cols = {c: c.lower() for c in df2.columns}
    df2.rename(columns=lower_cols, inplace=True)

    # Build Datetime
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

    # Guess columns for balance/equity
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
        .dropna(subset=["Datetime", "Equity"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    return out

# --- Compute month x year max relative DD table ---
def monthly_relative_dd_pivot(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["EquityPeak"] = tmp["Equity"].cummax()
    tmp["Rel_Drawdown_%"] = (tmp["Equity"] - tmp["EquityPeak"]) / tmp["EquityPeak"] * 100.0
    tmp["Year"] = tmp["Datetime"].dt.year
    tmp["Month"] = tmp["Datetime"].dt.month
    month_map = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Juin",7:"Juil",8:"Août",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
    tmp["MonthName"] = tmp["Month"].map(month_map)

    # min (most negative) per month
    m = tmp.groupby(["Year","Month","MonthName"])["Rel_Drawdown_%"].min().reset_index()

    # Pivot: Month rows x Year columns
    pivot = m.pivot_table(index=["Month","MonthName"], columns="Year", values="Rel_Drawdown_%", aggfunc="min").sort_index()
    pivot = pivot.reset_index().sort_values("Month").set_index("MonthName").drop(columns=["Month"])
    return pivot

# --- Styling (no type annotation to avoid pandas version issues) ---
def style_dd_table(pivot):
    if pivot.empty:
        return pivot.style

    # global min (most negative)
    min_val = float(np.nanmin(pivot.values))

    # highlight min in red
    def highlight_min(s):
        return [("font-weight:700; color:white; background-color:#B00020") if (v == min_val) else "" for v in s]

    styled = (
        pivot.style
        .format("{:.2f}%")
        .background_gradient(axis=None, cmap="Blues", vmin=np.nanmin(pivot.values), vmax=0.0)
        .apply(highlight_min, axis=1)
    )
    return styled

# --- UI ---
st.title("Max Relative Drawdown (Equity) — Tableau Mois × Année")
st.caption("Max drawdown relatif (Equity) pour chaque mois (lignes) et chaque année (colonnes). La valeur la plus faible (plus négative) est **mise en rouge**.")

uploaded = st.file_uploader("Dépose ton CSV MT5 (Date/Heure/Balance/Equity)", type=["csv"])

if uploaded is not None:
    try:
        df_raw = robust_read_csv(uploaded)
        uploaded.seek(0)
        df = normalize_columns(df_raw)

        if df.empty:
            st.error("Aucune donnée exploitable. Vérifie les colonnes Date/Heure/Balance/Equity.")
        else:
            pivot = monthly_relative_dd_pivot(df)

            st.subheader("Mois (lignes) × Années (colonnes) — Rel DD (%)")
            styled = style_dd_table(pivot)
            st.dataframe(styled, use_container_width=True)

            # Export CSV
            csv_bytes = pivot.to_csv().encode("utf-8")
            st.download_button("⬇️ Export CSV", data=csv_bytes, file_name="monthly_rel_dd_table.csv", mime="text/csv")

            if not pivot.empty:
                min_val = float(np.nanmin(pivot.values))
                pos = np.where(pivot.values == min_val)
                if len(pos[0]) > 0:
                    r, c = int(pos[0][0]), int(pos[1][0])
                    month = pivot.index[r]
                    year = pivot.columns[c]
                    st.info(f"📉 **Min global** : {min_val:.3f}% — **{month} {year}**")
    except Exception as e:
        st.exception(e)
else:
    st.info("Charge un CSV pour générer le tableau.")
