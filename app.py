import pandas as pd
import streamlit as st
import tennis_core as T

st.set_page_config(page_title="Tenisas: prematch analizatorius", layout="centered")
st.title("Tenisas: prematch analizatorius")

tour = st.sidebar.radio("Turas", ["ATP", "WTA"])
surface = st.sidebar.selectbox("Danga", ["Hard", "Clay", "Grass"])
if st.sidebar.button("Atnaujinti duomenis"):
    st.cache_data.clear()
    st.session_state.refresh = True


@st.cache_data(show_spinner="Keliami duomenys ir skaičiuojamas Elo...")
def prep(tour, refresh):
    df = T.load(tour, refresh)
    L = T.to_long(df)
    return df.date.max(), L, T.elo(df), L.player.value_counts().index.tolist()


try:
    last, L, elos, names = prep(tour, st.session_state.pop("refresh", False))
except Exception as e:
    st.error(str(e))
    st.stop()

today = pd.Timestamp.today().normalize()
st.caption(f"Naujausios rungtynės duomenyse: {last:%Y-%m-%d}")
st.caption("Duomenys: Jeff Sackmann / Tennis Abstract, licencija CC BY-NC-SA 4.0 (tik nekomerciniam naudojimui).")
if (today - last).days > 14:
    st.warning("Duomenys senesni nei 14 d. Naujausių rungtynių gali trūkti, formos aspektai gali būti netikslūs.")

st.sidebar.subheader("Svoriai (PP)")
W = {k: st.sidebar.number_input(lbl, 0.0, 100.0, float(w), 1.0, key="w_" + k) for k, lbl, _, w in T.FACTORS}

c1, c2 = st.columns(2)
p1 = c1.selectbox("Žaidėjas 1", names, index=None, placeholder="Rašyk vardą...")
p2 = c2.selectbox("Žaidėjas 2", names, index=None, placeholder="Rašyk vardą...")
if not p1 or not p2 or p1 == p2:
    st.info("Pasirink du skirtingus žaidėjus (rašyk pavardę, kad surastum).")
    st.stop()

sa, sb = T.stats(L, p1, surface, today, elos), T.stats(L, p2, surface, today, elos)
ca, cb, ncom = T.common(L, p1, p2, today)
sa["common"], sb["common"] = ca, cb

rows = [(lbl, mode, W[k], sa.get(k), sb.get(k)) for k, lbl, mode, _ in T.FACTORS]

st.subheader("Papildomi aspektai (ranka)")
extra = st.data_editor(
    pd.DataFrame({"Pavadinimas": pd.Series(dtype="str"), "Svoris": pd.Series(dtype="float"),
                  "Režimas": pd.Series(dtype="str"), "A": pd.Series(dtype="float"), "B": pd.Series(dtype="float")}),
    num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={"Režimas": st.column_config.SelectboxColumn(options=["pct", "hi"], help="pct = procentai, hi = daugiau geriau")})
for _, r in extra.dropna(subset=["Pavadinimas", "Svoris", "A", "B"]).iterrows():
    rows.append((r["Pavadinimas"], r["Režimas"] or "hi", r["Svoris"], r["A"], r["B"]))

A, B, table = T.score(rows)
if A + B == 0:
    st.error("Nepavyko gauti nė vieno aspekto. Patikrink žaidėjų vardus ir duomenų datą.")
    st.stop()
pa = A / (A + B) * 100

st.header(f"{p1} {pa:.0f}% – {100 - pa:.0f}% {p2}")
st.progress(int(round(pa)))
st.caption(f"PP: {A:.1f} : {B:.1f} iš {A + B:.1f}. Aspektai be duomenų praleidžiami, svoriai perskaičiuojami.")
wa, wb = T.h2h(L, p1, p2)
st.write(f"H2H per paskutinius 5 m.: {p1} {wa} – {wb} {p2}. Bendrų varžovų: {ncom}.")
st.dataframe(pd.DataFrame(table).rename(columns={"A": p1, "B": p2, "PP A": f"PP {p1}", "PP B": f"PP {p2}"}),
             use_container_width=True, hide_index=True)

st.subheader("Koeficientai (nebūtina)")
o1, o2 = st.columns(2)
q1 = o1.number_input(f"Koef. {p1}", 0.0, 100.0, 0.0, 0.01)
q2 = o2.number_input(f"Koef. {p2}", 0.0, 100.0, 0.0, 0.01)
if q1 > 1 and q2 > 1:
    for n, p, q in ((p1, pa / 100, q1), (p2, 1 - pa / 100, q2)):
        st.write(f"{n}: EV {(p * q - 1) * 100:+.1f}%")
    st.caption("PP procentai nėra sukalibruota tikimybė, todėl EV vertink atsargiai.")
