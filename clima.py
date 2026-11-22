from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

VERSION_APP = "2.3"
BASE_DIR = Path(__file__).resolve().parent
UMBRAL_FRIO_C = 12.0
SNOW_CODES = {71, 73, 75, 77, 85, 86}
STORM_CODES = {95, 96, 99}

st.set_page_config(
    page_title=f"Clima Bonito v{VERSION_APP} • App del Clima",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolver_ruta_csv(nombre_archivo):
    for ruta in [BASE_DIR / nombre_archivo, Path.cwd() / nombre_archivo, BASE_DIR / "data" / nombre_archivo, Path.cwd() / "data" / nombre_archivo]:
        if ruta.is_file():
            return ruta
    return next(BASE_DIR.rglob(nombre_archivo), None)


def _leer_csv_universal(ruta):
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(ruta, sep=None, engine="python", encoding=enc)
            df.columns = [str(c).strip().upper() for c in df.columns]
            if {"FECHA", "TMAX", "TMIN"}.issubset(df.columns):
                return df[["FECHA", "TMAX", "TMIN"]]
        except Exception:
            continue
    return None


@st.cache_resource(show_spinner=False)
def cargar_perfil_climatologico(nombre_archivo):
    ruta = _resolver_ruta_csv(nombre_archivo)
    df = _leer_csv_universal(ruta) if ruta else None
    if df is None or df.empty:
        return None

    fechas_str = df["FECHA"].astype(str).str.strip()
    dt = pd.to_datetime(fechas_str, format="%Y-%m-%d", errors="coerce")
    if dt.isna().mean() > 0.3:
        dt = pd.to_datetime(fechas_str, dayfirst=True, errors="coerce")

    tmax = pd.to_numeric(df["TMAX"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    tmin = pd.to_numeric(df["TMIN"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    valido = dt.notna() & tmax.notna() & tmin.notna() & tmax.between(-40, 60) & tmin.between(-45, 55)
    if not valido.any():
        return None

    doy_arr = dt[valido].dt.dayofyear.to_numpy(dtype=np.float64)
    anio_arr = dt[valido].dt.year.to_numpy(dtype=np.int32)
    tmax_arr = tmax[valido].to_numpy(dtype=np.float64)
    tmin_arr = tmin[valido].to_numpy(dtype=np.float64)
    peso_rec = np.where(anio_arr >= 2005, 1.5, np.where(anio_arr >= 1990, 1.2, 1.0))

    perfil = {}
    for d in range(1, 367):
        diff = ((doy_arr - d + 182.5) % 365.0) - 182.5
        mask = np.abs(diff) <= 10.0
        if not np.any(mask):
            mask = np.ones_like(diff, dtype=bool)
        w = (1.0 / (1.0 + (np.abs(diff[mask]) / 4.0) ** 2)) * peso_rec[mask]
        w_sum = float(w.sum())
        t_max_est = float(np.dot(tmax_arr[mask], w) / w_sum)
        t_min_est = float(np.dot(tmin_arr[mask], w) / w_sum)
        perfil[d] = (round(t_max_est, 2), round(t_min_est - 3.5 if t_min_est >= t_max_est else t_min_est, 2))
    return perfil


def es_ubicacion_hibrida(nombre_ciudad, lat, lon):
    n = (nombre_ciudad or "").strip().lower()
    if "saltillo" in n or (lat is not None and lon is not None and abs(lat - 25.4232) < 0.08 and abs(lon - (-101.0053)) < 0.08):
        return "saltillo"
    if "arteaga" in n or (lat is not None and lon is not None and abs(lat - 25.4450) < 0.08 and abs(lon - (-100.8480)) < 0.08):
        return "arteaga"
    return None


@st.cache_resource(show_spinner=False)
def obtener_sesion_http():
    s = requests.Session()
    s.headers.update({"User-Agent": f"ClimaBonito/{VERSION_APP}"})
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def obtener_clima_api(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,weathercode,is_day"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
            "relative_humidity_2m_mean,weathercode,uv_index_max"
            "&hourly=temperature_2m,precipitation_probability,relative_humidity_2m,weathercode"
            "&forecast_days=7&timezone=auto"
        )
        r = obtener_sesion_http().get(url, timeout=3.5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def obtener_sugerencias_ubicacion(query):
    if not query or len(query.strip()) < 2:
        return []
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query.strip()}&count=5&language=es&format=json"
        r = obtener_sesion_http().get(url, timeout=3.0)
        if r.status_code == 200 and "results" in r.json():
            return [
                {
                    "etiqueta": f"{res.get('name', '')}, {res.get('admin1', '')}, {res.get('country', '')}" if res.get("admin1") else f"{res.get('name', '')}, {res.get('country', '')}",
                    "lat": res.get("latitude"),
                    "lon": res.get("longitude"),
                    "nombre_corto": res.get("name", ""),
                    "estado": res.get("admin1", ""),
                }
                for res in r.json()["results"]
            ]
    except Exception:
        pass
    return []


def _ahora_mx():
    try:
        return datetime.now(ZoneInfo("America/Monterrey"))
    except Exception:
        return datetime.now()


def interpretar_weathercode(code, hora=12, minuto=0, is_day_api=None, temp_ref_c=None):
    es_noche = (int(is_day_api) == 0) if is_day_api is not None else ((hora * 60 + minuto) < 360 or (hora * 60 + minuto) >= 1110)

    # Nieve se evalúa primero para no interferir con Frío
    if code in SNOW_CODES:
        return ("Noche nevada", "❄️🌙") if es_noche else ("Nevado", "❄️")
    if code in STORM_CODES:
        return ("Tormenta eléctrica nocturna", "⚡🌙") if es_noche else ("Tormenta eléctrica", "⚡")
    if code is not None and 50 <= code < 90:
        return ("Noche lluviosa", "🌧️🌙") if es_noche else ("Lluvioso", "🌧️")
    # Frío cuando la temperatura de referencia (máxima del día o temperatura actual) es <= 12°C
    if temp_ref_c is not None and float(temp_ref_c) <= UMBRAL_FRIO_C:
        return ("Noche fría", "🥶🌙") if es_noche else ("Frío", "🥶")
    if code is None or code == 0:
        return ("Noche despejada", "🌙") if es_noche else ("Soleado", "☀️")
    if code in [1, 2]:
        return ("Noche parcialmente nublada", "☁️🌙") if es_noche else ("Parcialmente nublado", "⛅")
    if code in [3, 45, 48]:
        return ("Noche nublada", "☁️🌙") if es_noche else ("Nublado", "☁️")
    return ("Noche despejada", "🌙") if es_noche else ("Soleado", "☀️")


def obtener_condicion_actual_precisa(datos_api):
    ahora = _ahora_mx()
    h, m = ahora.hour, ahora.minute
    lluvia_actual = 0
    if datos_api and "hourly" in datos_api:
        probs = datos_api["hourly"].get("precipitation_probability", [])
        if h < len(probs) and probs[h] is not None:
            lluvia_actual = int(probs[h])

    if datos_api and "current" in datos_api:
        cur = datos_api["current"]
        temp_cur = float(cur.get("temperature_2m", 22.0))
        if float(cur.get("precipitation", 0) or 0) > 0 and lluvia_actual < 60:
            lluvia_actual = max(lluvia_actual, 70)
        cond, icono = interpretar_weathercode(cur.get("weathercode", 0), h, m, cur.get("is_day"), temp_ref_c=temp_cur)
        return cond, icono, temp_cur, lluvia_actual

    if datos_api and "hourly" in datos_api and datos_api["hourly"].get("temperature_2m"):
        idx_h = min(h, len(datos_api["hourly"]["temperature_2m"]) - 1)
        temp_h = float(datos_api["hourly"]["temperature_2m"][idx_h])
        code_h = datos_api["hourly"].get("weathercode", [0] * 24)[idx_h]
        cond, icono = interpretar_weathercode(code_h, h, m, temp_ref_c=temp_h)
        return cond, icono, temp_h, lluvia_actual

    es_noche = (h * 60 + m) < 360 or (h * 60 + m) >= 1110
    temp_fb = 9.0 if es_noche else 22.0
    cond_fb, icon_fb = interpretar_weathercode(0, h, m, temp_ref_c=temp_fb)
    return cond_fb, icon_fb, temp_fb, 0


for k, v in [("vista_detalle", False), ("vista_actualizaciones", False), ("dia_seleccionado", 0), ("unidades_temp", "°C"), ("easter_egg_activo", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

TEMAS_CLIMA_OSCURO = {
    "Soleado": {"bg": "#241B17", "card": "#362721", "accent": "#FFB703", "text": "#FFE6D5", "subtext": "#FFD1A3", "badge": "#FFB70333", "sidebar": "#2C201B", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Parcialmente nublado": {"bg": "#1C1D24", "card": "#282936", "accent": "#FFD166", "text": "#F4F5F6", "subtext": "#A0AAB2", "badge": "#FFD16633", "sidebar": "#22232D", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Nublado": {"bg": "#1A1726", "card": "#262138", "accent": "#FF80BF", "text": "#E8D7F1", "subtext": "#B5A1E8", "badge": "#FF80BF33", "sidebar": "#211C30", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Lluvioso": {"bg": "#131C26", "card": "#1E2B3B", "accent": "#38B6FF", "text": "#D6EEFF", "subtext": "#80CAFF", "badge": "#38B6FF33", "sidebar": "#182330", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Tormenta eléctrica": {"bg": "#0D1117", "card": "#161B22", "accent": "#F72585", "text": "#F8D7DA", "subtext": "#F28482", "badge": "#F7258533", "sidebar": "#090D12", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Frío": {"bg": "#0D1B2A", "card": "#162B44", "accent": "#4CC9F0", "text": "#E0F4FF", "subtext": "#8ECAE6", "badge": "#4CC9F033", "sidebar": "#09131E", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche fría": {"bg": "#07111E", "card": "#102238", "accent": "#56CFE1", "text": "#D8F3FF", "subtext": "#72EFDD", "badge": "#56CFE133", "sidebar": "#050B14", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Nevado": {"bg": "#162026", "card": "#202E38", "accent": "#90E0EF", "text": "#E0FBFC", "subtext": "#A5F0FC", "badge": "#90E0EF33", "sidebar": "#1B2730", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche despejada": {"bg": "#0B0E14", "card": "#151B26", "accent": "#A0C4FF", "text": "#E2E8F0", "subtext": "#8D99AE", "badge": "#A0C4FF33", "sidebar": "#0F141E", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche parcialmente nublada": {"bg": "#0E1320", "card": "#1A2130", "accent": "#B9FBC0", "text": "#E2E8F0", "subtext": "#94A3B8", "badge": "#B9FBC033", "sidebar": "#131926", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche nublada": {"bg": "#121622", "card": "#1E2436", "accent": "#BDB2FF", "text": "#ECE7FE", "subtext": "#9D8EC7", "badge": "#BDB2FF33", "sidebar": "#161B29", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche lluviosa": {"bg": "#0F172A", "card": "#1E293B", "accent": "#38BDF8", "text": "#E0F2FE", "subtext": "#7DD3FC", "badge": "#38BDF833", "sidebar": "#131F33", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Tormenta eléctrica nocturna": {"bg": "#0B0F19", "card": "#171E31", "accent": "#E879F9", "text": "#FAE8FF", "subtext": "#D946EF", "badge": "#E879F933", "sidebar": "#0D1322", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
    "Noche nevada": {"bg": "#111C24", "card": "#1C2D37", "accent": "#CAF0F8", "text": "#EDF6F9", "subtext": "#90E0EF", "badge": "#CAF0F833", "sidebar": "#14222B", "val_color": "#FFFFFF", "input_text": "#FFFFFF"},
}

with st.sidebar:
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 0px;'>🐾 Configuración (v{VERSION_APP})</h3><p style='text-align: center; font-size: 0.8rem; opacity: 0.7; margin-top: 0px;'>Personaliza tu aplicación</p><hr style='margin: 10px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; font-weight: 700; margin-bottom: 4px;'>📍 Seleccionar Ubicación:</p>", unsafe_allow_html=True)
    modo_ubicacion = st.radio("Modo de ubicación", ["Saltillo", "Arteaga", "Buscar otra ciudad"], label_visibility="collapsed")

    if modo_ubicacion == "Saltillo":
        municipio_limpio, estado_limpio, lat_actual, lon_actual = "Saltillo", "Coahuila", 25.4232, -101.0053
    elif modo_ubicacion == "Arteaga":
        municipio_limpio, estado_limpio, lat_actual, lon_actual = "Arteaga", "Coahuila", 25.4450, -100.8480
    else:
        busqueda_texto = st.text_input("Buscar otra ciudad:", placeholder="Ej. Monterrey, Guadalajara...", label_visibility="collapsed")
        sugerencias = obtener_sugerencias_ubicacion(busqueda_texto)
        if sugerencias:
            opciones_map = {s["etiqueta"]: s for s in sugerencias}
            elegida = opciones_map[st.selectbox("Sugerencias", list(opciones_map.keys()), label_visibility="collapsed")]
            municipio_limpio, estado_limpio, lat_actual, lon_actual = elegida["nombre_corto"], elegida["estado"], elegida["lat"], elegida["lon"]
        else:
            municipio_limpio, estado_limpio, lat_actual, lon_actual = "Saltillo", "Coahuila", 25.4232, -101.0053

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(f"🆕 Novedades v{VERSION_APP}", use_container_width=True):
        st.session_state.vista_actualizaciones = True
        st.session_state.vista_detalle = False
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    datos_api_temporal = obtener_clima_api(lat_actual, lon_actual)
    clima_hoy, _, _, _ = obtener_condicion_actual_precisa(datos_api_temporal)
    cfg = TEMAS_CLIMA_OSCURO.get(clima_hoy, TEMAS_CLIMA_OSCURO["Nublado"])

    components.html(f"""
    <div style="background: {cfg['card']}; border: 2.5px dashed {cfg['accent']}; border-radius: 20px; padding: 12px; text-align: center; font-family: 'Quicksand', sans-serif; box-shadow: 0 4px 12px {cfg['accent']}33;">
        <div style="font-size: 1.5rem; font-weight: 700; color: {cfg['val_color']}; display: flex; align-items: center; justify-content: center; gap: 6px;">
            <span id="relojHora">00:00:00</span>
            <span id="relojAmpm" style="font-size: 0.85rem; color: {cfg['accent']};">AM</span>
        </div>
        <div id="relojFecha" style="font-size: 0.75rem; color: {cfg['subtext']}; font-weight: 600; margin-top: 4px;"></div>
    </div>
    <script>
        function actualizarReloj() {{
            const ahora = new Date();
            const partes = new Intl.DateTimeFormat('en-US', {{ timeZone: 'America/Monterrey', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }}).formatToParts(ahora);
            let h = '00', m = '00', s = '00';
            partes.forEach(p => {{ if (p.type === 'hour') h = p.value; if (p.type === 'minute') m = p.value; if (p.type === 'second') s = p.value; }});
            let horaInt = parseInt(h, 10);
            document.getElementById('relojAmpm').innerText = horaInt >= 12 ? 'PM' : 'AM';
            horaInt = (horaInt % 12) || 12;
            document.getElementById('relojHora').innerText = String(horaInt).padStart(2, '0') + ':' + m + ':' + s;
            document.getElementById('relojFecha').innerText = new Intl.DateTimeFormat('es-ES', {{ timeZone: 'America/Monterrey', weekday: 'short', day: 'numeric', month: 'short' }}).format(ahora);
        }}
        setInterval(actualizarReloj, 1000); actualizarReloj();
    </script>
    """, height=95)

    st.markdown("<p style='font-size: 0.85rem; font-weight: 700; margin-bottom: 2px;'>🌡️ Unidades de Temperatura:</p>", unsafe_allow_html=True)
    unidades_sel = st.selectbox("Unidades", ("°C", "°F"), label_visibility="collapsed")
    if unidades_sel != st.session_state.unidades_temp:
        st.session_state.unidades_temp = unidades_sel
        st.rerun()

    st.markdown("<br><p style='font-size: 0.85rem; font-weight: 700; margin-bottom: 2px;'>🎶 Radio Lo-Fi (Inicia al abrir):</p>", unsafe_allow_html=True)
    components.html(f"""
    <div style="background: {cfg['card']}; border: 1.5px solid {cfg['accent']}66; border-radius: 16px; padding: 12px; font-family: 'Quicksand', sans-serif; color: {cfg['text']}; box-shadow: 0 4px 12px {cfg['accent']}22;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; font-weight: 700; color: {cfg['accent']};">🎧 Chill Beats 24/7</span>
            <span id="statusAudio" style="font-size: 0.7rem; opacity: 0.8; color: #38B6FF;">Reproduciendo</span>
        </div>
        <audio id="audioLofi" preload="none" loop autoplay><source src="https://stream.zeno.fm/f3wvbbqmdg8uv" type="audio/mp4"></audio>
        <button onclick="togglePlay()" id="btnPlay" style="background: {cfg['accent']}; color: #222; border: none; border-radius: 10px; padding: 7px; font-weight: bold; cursor: pointer; font-size: 0.8rem; width: 100%;">⏸️ Pausa</button>
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
            <span style="font-size: 0.7rem;">🔈</span>
            <input type="range" min="0" max="1" step="0.01" value="0.06" oninput="document.getElementById('audioLofi').volume = this.value" style="width: 100%; accent-color: {cfg['accent']}; cursor: pointer;">
            <span style="font-size: 0.7rem;">🔊</span>
        </div>
    </div>
    <script>
        const audio = document.getElementById('audioLofi'); audio.volume = 0.06;
        audio.play().catch(() => {{ document.getElementById('statusAudio').innerText = "Pulsa Play"; document.getElementById('btnPlay').innerHTML = "▶️ Reproducir"; }});
        function togglePlay() {{
            const btn = document.getElementById('btnPlay'), stt = document.getElementById('statusAudio');
            if (audio.paused) {{ audio.play().then(() => {{ btn.innerHTML = "⏸️ Pausa"; stt.innerText = "Reproduciendo"; }}).catch(() => {{}}); }}
            else {{ audio.pause(); btn.innerHTML = "▶️ Reproducir"; stt.innerText = "En pausa"; }}
        }}
    </script>
    """, height=130)

    st.markdown(f"""
        <br><div style="background: {cfg['card']}; border: 1px dashed {cfg['accent']}88; border-radius: 12px; padding: 8px 10px; text-align: center; margin-bottom: 8px;">
            <span style="font-size: 0.75rem; color: {cfg['subtext']}; font-weight: 600; line-height: 1.2; display: block;">
                💖 Pica con mucho amor el botón <b style="color: {cfg['accent']};">"Ekuipo"</b> para desbloquear una pequeña sorpresa mágica ✨🐾
            </span>
        </div>
    """, unsafe_allow_html=True)

    if st.button("✨ Ekuipo", use_container_width=True, key="btn_ekuipo_magico"):
        st.session_state.easter_egg_activo = not st.session_state.easter_egg_activo
    if st.session_state.easter_egg_activo:
        st.balloons()
        st.success("Ekuipo conformado por Ari y Angela 🐾✨")


def convertir_temp(celsius):
    return round((celsius * 9 / 5) + 32, 1) if st.session_state.unidades_temp == "°F" else round(celsius, 1)


def obtener_gatito_segun_clima(t_c, uv_val=5.0, prob_lluvia=0, card_id=0):
    prob_int = int(prob_lluvia or 0)
    es_lluvia_o_tormenta = ("lluvioso" in t_c.lower()) or ("tormenta" in t_c.lower())
    extra_cat_class = ""

    if prob_int >= 50:
        if "tormenta" in t_c.lower():
            frases = [f"Truenos y rayos ({prob_int}% lluvia), un chocolatito caliente ⚡☕", f"Tormenta activa con {prob_int}% de lluvia, quédate a salvo ⚡🌧️", f"¡Rayitos en el cielo! {prob_int}% de probabilidad de lluvia ⚡🐾"]
        else:
            frases = [f"Escucha las gotitas caer ({prob_int}% lluvia) ☂️🌧️", f"Día lluvioso con {prob_int}% de probabilidad, ¡no olvides tu paraguas! 🌧️✨", f"Gotitas lindas cayendo ({prob_int}%), ¡qué buen plan bajo techo! ☂️🐾"]
        ojos_html, accesorio_html = '<div class="ojitos"><span></span><span></span></div>', '<div class="icono-clima-flotante">☂️</div>'
    elif t_c == "Noche fría":
        frases = ["¡Brrr, noche fría (≤12°C)! Saca tu cobija más calientita 🥶🌙", "Aire heladito de madrugada, ¡no te descubras! 🌬️💤", "Noche fresca (≤12°C), ideal para un tecito caliente ☕🌙"]
        ojos_html, accesorio_html, extra_cat_class = '<div class="ojitos-frio"><span></span><span></span></div>', '<div class="icono-clima-flotante">🥶🌙</div>', " gato-noche-fria-estilo"
    elif t_c == "Frío":
        frases = ["¡Brrr, qué frío (≤12°C)! Saca tu suéter favorito 🧥🥶", "Aire heladito hoy, ¡abrígate bien antes de salir! 🌬️🧣", "Máxima de 12°C o menos, ¡qué rico un chocolatito calientito! ☕🥶"]
        ojos_html, accesorio_html, extra_cat_class = '<div class="ojitos-frio"><span></span><span></span></div>', '<div class="icono-clima-flotante">🥶</div>', " gato-frio-estilo"
    elif t_c in ("Nevado", "Noche nevada"):
        frases = ["Copitos cayendo sobre el paisaje ❄️✨", "Día nevado y mágico, ¡cuídate del hielo! ❄️🐾", "Paisaje blanco y sereno por la nieve ❄️☃️"]
        ojos_html, accesorio_html, extra_cat_class = '<div class="ojitos-nevado"><span></span><span></span></div>', '<div class="icono-clima-flotante">❄️</div>', " gato-nevado-estilo"
    elif t_c == "Soleado":
        uv_txt = f" (UV {uv_val})" if uv_val >= 8 else ""
        frases = [f"¡Solcito hermoso! Usa tus lentes de sol{uv_txt} 🕶️✨", f"¡Día radiante! Toma agüita y disfruta el sol{uv_txt} ☀️🐾", f"¡Qué brillo tan bonito! No olvides protector solar{uv_txt} ☀️😎"]
        ojos_html, accesorio_html = '<div class="lentes-css-ojos"><div class="cristal izq"></div><div class="puente"></div><div class="cristal der"></div></div>', '<div class="icono-clima-flotante">☀️</div>'
    elif "Parcialmente nublado" in t_c:
        frases = ["Ratos de sol y nubes ~ ⛅", "Nubes coquetas dejando pasar el solcito ⛅✨", "Clima equilibrado entre nubes y luz ~ ⛅🐾"]
        ojos_html, accesorio_html = '<div class="ojitos"><span></span><span></span></div>', '<div class="icono-clima-flotante">⛅</div>'
    elif "Noche parcialmente nublada" in t_c:
        frases = ["Noche fresca con nubes pasajeras ☁️🌙", "Nubes ligeras paseando bajo la luna ☁️✨", "Brisa nocturna tranquila entre nubes ~ 🌙🐾"]
        ojos_html, accesorio_html = '<div class="ojitos-durmiendo"><span></span><span></span></div>', '<div class="icono-clima-flotante">☁️</div>'
    elif "Noche despejada" in t_c:
        frases = ["¡Buenas noches! Estrellas brillando 🌙✨", "Cielo despejado y estrellitas para soñar bonito 🌙⭐", "Noche serena bajo la luz de la luna ~ 🌙🐾"]
        ojos_html, accesorio_html = '<div class="ojitos-durmiendo"><span></span><span></span></div>', '<div class="icono-clima-flotante">🌙</div>'
    elif "Noche nublada" in t_c:
        frases = ["Noche nubladita y calmada ~ ☁️🌙", "Cielo cubierto y descanso suavecito ☁️💤", "Noche apacible con nubes en calma ☁️🌙"]
        ojos_html, accesorio_html = '<div class="ojitos-durmiendo"><span></span><span></span></div>', '<div class="icono-clima-flotante">☁️</div>'
    elif "Noche lluviosa" in t_c or (es_lluvia_o_tormenta and "noche" in t_c.lower()):
        frases = ["Lluvia suave de madrugada 🌧️🌙", "Gotitas nocturnas arrullando la noche 🌧️✨", "Sonido de lluvia tranquila en la oscuridad 🌧️🐾"]
        ojos_html, accesorio_html = '<div class="ojitos-durmiendo"><span></span><span></span></div>', '<div class="icono-clima-flotante">🌧️</div>'
    elif "Tormenta eléctrica" in t_c:
        frases = ["Rayos iluminando la noche ⚡🌙", "Destellos en el cielo, ¡cuídate! ⚡✨", "Truenos a lo lejos, quédate a salvo ⚡🐾"]
        ojos_html, accesorio_html = '<div class="ojitos"><span></span><span></span></div>', '<div class="icono-clima-flotante">⚡</div>'
    else:
        frases = ["Día nubladito y fresco ~ ☁️", "Cielo grisecito ideal para ir tranqui ☁️🐾", "Nubes suaves acompañando el día ~ ☁️✨"]
        ojos_html, accesorio_html = '<div class="ojitos-durmiendo"><span></span><span></span></div>', '<div class="icono-clima-flotante">☁️</div>'

    frase_id = f"frase_cat_{card_id}_{abs(hash(t_c)) % 10000}"
    return f"""
        <div class="tarjeta-gatito-box">
            <div class="gato-avatar{extra_cat_class}">
                <div class="oreja izq"></div><div class="oreja der"></div>
                <div class="cara">{ojos_html}<div class="hocico">ω</div></div>
                {accesorio_html}
            </div>
            <div class="burbuja-texto"><span id="{frase_id}" class="frase-cat">{frases[0]}</span></div>
        </div>
        <script>
            (function() {{
                const frases = {json.dumps(frases, ensure_ascii=False)};
                const el = document.getElementById("{frase_id}");
                if (!el) return;
                let idx = 0;
                setInterval(() => {{ idx = (idx + 1) % frases.length; el.innerText = frases[idx]; }}, 3500);
            }})();
        </script>
    """


def generar_pronostico_inteligente(nombre_ciudad, lat, lon):
    datos_api = obtener_clima_api(lat, lon)
    hoy = _ahora_mx()
    hibrido_tipo = es_ubicacion_hibrida(nombre_ciudad, lat, lon)
    perfil_hist = cargar_perfil_climatologico("datos_limpios_salt.csv" if hibrido_tipo == "saltillo" else "datos_limpios_arte.csv") if hibrido_tipo else None

    meses_es = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
    cond_h, icono_h, temp_cur_c, lluvia_cur = obtener_condicion_actual_precisa(datos_api)
    predicciones = []

    for i in range(7):
        d_obj = hoy + timedelta(days=i)
        fecha_str = f"{d_obj.day} de {meses_es[d_obj.month]}"
        etiqueta = f"Hoy • {fecha_str}" if i == 0 else (f"Mañana • {fecha_str}" if i == 1 else fecha_str)

        tmax_c, tmin_c, hum, lluvia, uv_val, w_code = 24.0, 12.0, 45, 0, 5.0, 0
        if datos_api and "daily" in datos_api:
            daily = datos_api["daily"]
            tmax_c = float(daily["temperature_2m_max"][i])
            tmin_c = float(daily["temperature_2m_min"][i])
            lluvia = int(daily.get("precipitation_probability_max", [0] * 7)[i] or 0)
            hum = int(daily.get("relative_humidity_2m_mean", [50] * 7)[i] or 50)
            uv_val = float(daily.get("uv_index_max", [5.0] * 7)[i] or 5.0)
            w_code = daily.get("weathercode", [0] * 7)[i]

        if hibrido_tipo and perfil_hist is not None:
            tmax_hist, tmin_hist = perfil_hist.get(d_obj.timetuple().tm_yday, (tmax_c, tmin_c))
            peso_api = max(0.60, 0.85 - 0.04 * i)
            tmax_c = peso_api * tmax_c + (1.0 - peso_api) * tmax_hist
            tmin_c = peso_api * tmin_c + (1.0 - peso_api) * tmin_hist
            if i == 0:
                tmax_c, tmin_c = max(tmax_c, temp_cur_c), min(tmin_c, temp_cur_c)
            if tmin_c >= tmax_c:
                tmin_c = tmax_c - 3.5

        if i == 0 and datos_api:
            t_c, icono, lluvia = cond_h, icono_h, max(lluvia, lluvia_cur)
        else:
            # Usa la temperatura máxima del día para evaluar Frío (<= 12°C) sin interferir con Nevado
            t_c, icono = interpretar_weathercode(w_code, 12, 0, temp_ref_c=tmax_c)

        predicciones.append({
            "ID": i,
            "FechaTitulo": etiqueta,
            "TMAX": convertir_temp(tmax_c),
            "TMIN": convertir_temp(tmin_c),
            "Humedad": hum,
            "Lluvia": lluvia,
            "UV": round(uv_val, 1),
            "Tipo": t_c,
            "Icono": icono,
            "GatitoHtml": obtener_gatito_segun_clima(t_c, uv_val, lluvia, card_id=i),
        })

    return pd.DataFrame(predicciones), datos_api, convertir_temp(temp_cur_c), lluvia_cur


df_p, datos_api_raw, temp_actual_val, lluvia_actual_val = generar_pronostico_inteligente(municipio_limpio, lat_actual, lon_actual)
cond_actual_real = df_p.iloc[0]["Tipo"] if not df_p.empty else "Soleado"
icono_actual_real = df_p.iloc[0]["Icono"] if not df_p.empty else "☀️"


def generar_html_animacion_fondo(tipo_clima):
    tc = tipo_clima.lower()
    if tc in ("frío", "frio"):
        ondas = "".join([f'<div class="rafaga-frio" style="top: {18 + i*22}%; animation-delay: {i*0.9}s;">🌬️</div>' for i in range(3)])
        return f'<div class="clima-anim-box"><div class="aura-frio-intenso"></div><div class="escarcha-esquina">🥶</div>{ondas}</div>'
    if tc in ("noche fría", "noche fria"):
        ondas = "".join([f'<div class="rafaga-frio-noche" style="top: {16 + i*24}%; animation-delay: {i*1.1}s;">🌬️</div>' for i in range(3)])
        estrellas = "".join([f'<div class="estrella-noche" style="left: {(i*19)%92}%; top: {(i*21)%82}%; animation-delay: {(i*0.35)%2}s;">✨</div>' for i in range(5)])
        return f'<div class="clima-anim-box"><div class="aura-noche-fria"></div><div class="luna-noche">🌙</div>{estrellas}{ondas}</div>'
    if "noche despejada" in tc:
        estrellas = "".join([f'<div class="estrella-noche" style="left: {(i*18)%95}%; top: {(i*22)%85}%; animation-delay: {(i*0.3)%2}s;">✨</div>' for i in range(6)])
        return f'<div class="clima-anim-box"><div class="cometa-veloz-bg"></div><div class="luna-noche">🌙</div>{estrellas}</div>'
    if "noche parcialmente" in tc:
        return '<div class="clima-anim-box"><div class="luna-noche" style="opacity: 0.8;">🌙</div><div class="nube-desplazandose" style="top: 22%; animation-duration: 16s;">☁️</div></div>'
    if "noche nublada" in tc:
        nubes = "".join([f'<div class="nube-desplazandose-lenta" style="top: {10 + i*25}%; animation-duration: {12 + i*3}s; animation-delay: {i*1.2}s;">☁️</div>' for i in range(2)])
        return f'<div class="clima-anim-box"><div class="nebulosa-nocturna"></div>{nubes}</div>'
    if "noche lluviosa" in tc:
        gotas = "".join([f'<div class="gota-noche" style="left: {i*16}%; animation-delay: {(i*0.1)%1}s;"></div>' for i in range(6)])
        return f'<div class="clima-anim-box"><div class="luna-noche" style="opacity: 0.6;">🌙</div>{gotas}<div class="radar-nocturno"></div></div>'
    if "tormenta eléctrica nocturna" in tc:
        gotas = "".join([f'<div class="gota-noche-intensa" style="left: {i*16}%; animation-delay: {(i*0.1)%1}s;"></div>' for i in range(6)])
        return f'<div class="clima-anim-box"><div class="flash-rayo-pantalla-noche"></div><div class="luna-noche" style="opacity: 0.4;">🌙</div>{gotas}</div>'
    if "noche nevada" in tc:
        copos = "".join([f'<div class="copo-noche" style="left: {i*20}%; animation-delay: {(i*0.2)%2}s;">❄️</div>' for i in range(4)])
        return f'<div class="clima-anim-box"><div class="luna-noche">🌙</div><div class="aura-fria"></div>{copos}</div>'
    if "soleado" in tc:
        particulas = "".join([f'<div class="particula-destello-sol" style="left: {(i*25)%92}%; top: {(i*20)%78}%; animation-delay: {(i*0.4)%3}s;">✨</div>' for i in range(4)])
        return f'<div class="clima-anim-box"><div class="sol-dia-anim">☀️</div>{particulas}</div>'
    if "parcialmente nublado" in tc:
        return '<div class="clima-anim-box"><div class="sol-dia-anim" style="opacity: 0.85;">🌤️</div><div class="nube-desplazandose" style="top: 22%; animation-duration: 14s;">☁️</div></div>'
    if "nublado" in tc:
        nubes = "".join([f'<div class="nube-desplazandose" style="top: {10 + i*22}%; animation-duration: {11 + i*2}s; animation-delay: {i*1.1}s;">☁️</div>' for i in range(2)])
        return f'<div class="clima-anim-box">{nubes}</div>'
    if "lluvioso" in tc:
        gotas = "".join([f'<div class="gota-dia" style="left: {i*16}%; animation-delay: {(i*0.1)%1}s;"></div>' for i in range(6)])
        return f'<div class="clima-anim-box">{gotas}</div>'
    if "tormenta eléctrica" in tc:
        gotas = "".join([f'<div class="gota-dia-intensa" style="left: {i*16}%; animation-delay: {(i*0.1)%1}s;"></div>' for i in range(6)])
        return f'<div class="clima-anim-box"><div class="flash-rayo-pantalla-dia"></div>{gotas}</div>'
    if "nevado" in tc:
        copos = "".join([f'<div class="copo-dia" style="left: {i*20}%; animation-delay: {(i*0.2)%2}s;">❄️</div>' for i in range(4)])
        return f'<div class="clima-anim-box">{copos}</div>'
    return '<div class="clima-anim-box"><div class="sol-dia-anim">☀️</div></div>'


cfg = TEMAS_CLIMA_OSCURO.get(cond_actual_real, TEMAS_CLIMA_OSCURO["Nublado"])
hex_bg = cfg["bg"].lstrip("#")
r, g, b = int(hex_bg[0:2], 16), int(hex_bg[2:4], 16), int(hex_bg[4:6], 16)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&display=swap');
    .stApp {{ background-color: rgba({r}, {g}, {b}, 0.98) !important; color: {cfg['text']} !important; font-family: 'Quicksand', sans-serif; }}
    @media (max-width: 768px) {{ .row-widget.stHorizontal {{ flex-direction: column !important; }} div[data-testid="column"] {{ width: 100% !important; flex: 100% !important; min-width: 100% !important; margin-bottom: 12px; }} }}
    .clima-anim-box {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0; overflow: hidden; margin: 0; padding: 0; }}
    @keyframes soploHelado {{ 0% {{ transform: translateX(-18vw) translateY(0px); opacity: 0; }} 20% {{ opacity: 0.78; }} 80% {{ opacity: 0.78; }} 100% {{ transform: translateX(112vw) translateY(-18px); opacity: 0; }} }}
    .rafaga-frio {{ position: absolute; font-size: 2.3rem; animation: soploHelado 6.5s ease-in-out infinite; filter: drop-shadow(0 0 10px #4CC9F0); will-change: transform; }}
    .rafaga-frio-noche {{ position: absolute; font-size: 2.2rem; animation: soploHelado 7.5s ease-in-out infinite; filter: drop-shadow(0 0 12px #56CFE1); will-change: transform; }}
    .escarcha-esquina {{ position: absolute; top: 8%; right: 9%; font-size: 4.4rem; animation: brilloEstatico 3.5s ease-in-out infinite; }}
    .aura-frio-intenso {{ position: absolute; width: 100vw; height: 100vh; top: 0; left: 0; background: radial-gradient(circle at 80% 15%, rgba(76, 201, 240, 0.16) 0%, rgba(13, 27, 42, 0.05) 65%); pointer-events: none; }}
    .aura-noche-fria {{ position: absolute; width: 100vw; height: 100vh; top: 0; left: 0; background: radial-gradient(circle at 80% 15%, rgba(86, 207, 225, 0.14) 0%, rgba(7, 17, 30, 0.08) 70%); pointer-events: none; }}
    @keyframes flashDia {{ 0%, 85%, 93%, 100% {{ background-color: rgba(255, 255, 255, 0); }} 88%, 91% {{ background-color: rgba(255, 255, 255, 0.3); }} }}
    @keyframes flashNoche {{ 0%, 85%, 93%, 100% {{ background-color: rgba(0, 0, 0, 0); }} 88%, 91% {{ background-color: rgba(232, 121, 249, 0.28); }} }}
    .flash-rayo-pantalla-dia {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0; animation: flashDia 3s infinite; }}
    .flash-rayo-pantalla-noche {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0; animation: flashNoche 2.5s infinite; }}
    @keyframes moverNubeFondo {{ 0% {{ transform: translateX(-25vw); opacity: 0; }} 15% {{ opacity: 0.85; }} 85% {{ opacity: 0.85; }} 100% {{ transform: translateX(115vw); opacity: 0; }} }}
    .nube-desplazandose {{ position: absolute; font-size: 4.5rem; animation: moverNubeFondo linear infinite; filter: drop-shadow(0 6px 15px rgba(0,0,0,0.25)); z-index: 0; will-change: transform; }}
    .nube-desplazandose-lenta {{ position: absolute; font-size: 4.8rem; animation: moverNubeFondo linear infinite; filter: drop-shadow(0 6px 20px rgba(189,178,255,0.2)); z-index: 0; opacity: 0.9; will-change: transform; }}
    @keyframes brilloEstatico {{ 0%, 100% {{ transform: scale(1); filter: drop-shadow(0 0 15px {cfg['accent']}); }} 50% {{ transform: scale(1.06); filter: drop-shadow(0 0 32px {cfg['accent']}); }} }}
    .sol-dia-anim, .luna-noche {{ position: absolute; top: 7%; right: 9%; font-size: 4.5rem; animation: brilloEstatico 4s ease-in-out infinite; }}
    @keyframes destelloParticula {{ 0%, 100% {{ opacity: 0.2; transform: scale(0.7); }} 50% {{ opacity: 0.9; transform: scale(1.3); filter: drop-shadow(0 0 10px #FFB703); }} }}
    .particula-destello-sol {{ position: absolute; font-size: 1.4rem; animation: destelloParticula 2.5s ease-in-out infinite; }}
    @keyframes titilarEstrella {{ 0%, 100% {{ opacity: 0.2; transform: scale(0.7); }} 50% {{ opacity: 1; transform: scale(1.3); filter: drop-shadow(0 0 8px #A0C4FF); }} }}
    .estrella-noche {{ position: absolute; font-size: 1.2rem; animation: titilarEstrella 2s ease-in-out infinite; }}
    @keyframes cometaCruzando {{ 0% {{ transform: translateX(-10vw) translateY(-5vh); opacity: 0; }} 20% {{ opacity: 1; }} 100% {{ transform: translateX(110vw) translateY(55vh); opacity: 0; }} }}
    .cometa-veloz-bg {{ position: absolute; top: 5%; left: -5%; width: 120px; height: 2px; background: linear-gradient(90deg, #FFFFFF, rgba(160,196,255,0)); animation: cometaCruzando 7s linear infinite; transform: rotate(-25deg); }}
    @keyframes caidaGota {{ 0% {{ transform: translateY(-20px); opacity: 0; }} 30% {{ opacity: 0.9; }} 100% {{ transform: translateY(100vh); opacity: 0.2; }} }}
    .gota-dia {{ position: absolute; width: 2px; height: 18px; background: {cfg['accent']}; animation: caidaGota 1.2s linear infinite; will-change: transform; }}
    .gota-dia-intensa {{ position: absolute; width: 2.5px; height: 22px; background: {cfg['accent']}; animation: caidaGota 0.7s linear infinite; will-change: transform; }}
    .gota-noche {{ position: absolute; width: 2px; height: 18px; background: #38BDF8; animation: caidaGota 1.2s linear infinite; will-change: transform; }}
    .gota-noche-intensa {{ position: absolute; width: 2.5px; height: 22px; background: #E879F9; animation: caidaGota 0.7s linear infinite; will-change: transform; }}
    @keyframes caidaCopo {{ 0% {{ transform: translateY(-20px) rotate(0deg); opacity: 0; }} 30% {{ opacity: 0.9; }} 100% {{ transform: translateY(100vh) rotate(360deg); opacity: 0.2; }} }}
    .copo-dia, .copo-noche {{ position: absolute; font-size: 1.2rem; animation: caidaCopo 3s linear infinite; will-change: transform; }}
    @keyframes radarPulso {{ 0% {{ transform: scale(0.8); opacity: 0.05; }} 50% {{ transform: scale(1.3); opacity: 0.25; }} 100% {{ transform: scale(0.8); opacity: 0.05; }} }}
    .radar-nocturno {{ position: absolute; width: 500px; height: 500px; bottom: -150px; left: -150px; background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, rgba(0,0,0,0) 70%); border-radius: 50%; animation: radarPulso 4s ease-in-out infinite; }}
    .nebulosa-nocturna {{ position: absolute; width: 350px; height: 350px; top: -80px; right: -80px; background: radial-gradient(circle, rgba(189,178,255,0.2) 0%, rgba(0,0,0,0) 70%); border-radius: 50%; }}
    .aura-fria {{ position: absolute; width: 100vw; height: 100vh; top: 0; left: 0; background: radial-gradient(circle at 80% 20%, rgba(144,224,239,0.1) 0%, rgba(0,0,0,0) 70%); pointer-events: none; }}
    section[data-testid="stSidebar"] {{ background-color: {cfg['sidebar']} !important; border-right: 1.5px solid {cfg['accent']}33; }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div {{ color: {cfg['text']} !important; }}
    .stTextInput input, .stSelectbox select, div[data-baseweb="select"] > div {{ color: {cfg['input_text']} !important; background-color: {cfg['card']} !important; border-color: {cfg['accent']}66 !important; }}
    div[data-testid="stVerticalBlock"] {{ position: relative; z-index: 1; }}
    @keyframes saltitoGatito {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-4px); }} }}
    .tarjeta-gatito-box {{ background: {cfg['badge']}; border: 1.5px dashed {cfg['accent']}cc; border-radius: 16px; padding: 10px; text-align: center; margin: 10px 0 6px 0; animation: saltitoGatito 3s ease-in-out infinite; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 110px; }}
    .gato-avatar {{ position: relative; width: 52px; height: 40px; background: {cfg['accent']}; border-radius: 24px 24px 16px 16px; margin: 4px auto; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }}
    .gato-avatar.gato-frio-estilo {{ background: linear-gradient(180deg, #4CC9F0 0%, #3A86FF 100%); border: 1.5px solid #CAF0F8; }}
    .gato-avatar.gato-frio-estilo .oreja {{ border-bottom-color: #4CC9F0; }}
    .gato-avatar.gato-noche-fria-estilo {{ background: linear-gradient(180deg, #56CFE1 0%, #277DA1 100%); border: 1.5px solid #A9DEF9; }}
    .gato-avatar.gato-noche-fria-estilo .oreja {{ border-bottom-color: #56CFE1; }}
    .gato-avatar.gato-nevado-estilo {{ background: linear-gradient(180deg, #E0FBFC 0%, #90E0EF 100%); border: 1.5px solid #FFFFFF; }}
    .gato-avatar.gato-nevado-estilo .oreja {{ border-bottom-color: #E0FBFC; }}
    .gato-avatar .oreja {{ position: absolute; top: -8px; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-bottom: 13px solid {cfg['accent']}; }}
    .gato-avatar .oreja.izq {{ left: 3px; transform: rotate(-12deg); }}
    .gato-avatar .oreja.der {{ right: 3px; transform: rotate(12deg); }}
    .cara {{ position: relative; padding-top: 10px; }}
    .ojitos span {{ display: inline-block; width: 5px; height: 5px; background: #FFFFFF; border-radius: 50%; margin: 0 5px; }}
    .ojitos-frio span {{ display: inline-block; width: 6px; height: 6px; background: #FFFFFF; border: 1.5px solid #0D1B2A; border-radius: 50%; margin: 0 4px; }}
    .ojitos-nevado span {{ display: inline-block; width: 5px; height: 5px; background: #1B2730; border-radius: 50%; margin: 0 5px; }}
    .ojitos-durmiendo span {{ display: inline-block; width: 6px; height: 2px; background: #FFFFFF; border-radius: 2px; margin: 0 4px; }}
    .lentes-css-ojos {{ display: flex; align-items: center; justify-content: center; gap: 2px; margin: 1px auto 0 auto; }}
    .lentes-css-ojos .cristal {{ width: 14px; height: 9px; background: #111111; border-radius: 3px; }}
    .lentes-css-ojos .puente {{ width: 3px; height: 2px; background: #111111; }}
    .hocico {{ font-size: 0.65rem; color: #FFFFFF; font-weight: bold; line-height: 0.7; margin-top: 3px; }}
    .icono-clima-flotante {{ position: absolute; top: -8px; right: -10px; font-size: 1.15rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); }}
    .burbuja-texto {{ background: {cfg['card']}; border: 1px solid {cfg['accent']}44; border-radius: 10px; padding: 5px 8px; margin-top: 6px; width: 100%; min-height: 36px; display: flex; align-items: center; justify-content: center; }}
    .frase-cat {{ font-size: 0.73rem; color: {cfg['accent']}; font-weight: 700; display: block; line-height: 1.25; }}
    .tarjeta-clima-custom {{ background: {cfg['card']}; border: 2px solid {cfg['accent']}66; border-radius: 20px; padding: 16px 10px; text-align: center; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); margin-bottom: 10px; transition: transform 0.3s ease, border-color 0.3s ease; }}
    .tarjeta-clima-custom:hover {{ transform: translateY(-8px); border-color: {cfg['accent']}; }}
    .fecha-tarjeta {{ color: {cfg['accent']}; font-size: 0.95rem; font-weight: 700; margin-bottom: 6px; }}
    .temp-tarjeta {{ color: {cfg['val_color']}; font-size: 1.9rem; font-weight: 700; line-height: 1.1; margin-bottom: 6px; }}
    .min-tarjeta {{ background-color: {cfg['badge']}; color: {cfg['text']}; border-radius: 10px; padding: 3px 10px; display: inline-block; font-size: 0.78rem; font-weight: 700; border: 1px solid {cfg['accent']}44; margin-bottom: 4px; }}
    .extra-detalles {{ font-size: 0.73rem; color: {cfg['subtext']}; font-weight: 600; margin-top: 4px; }}
    .lluvia-badge {{ display: inline-block; margin-top: 5px; padding: 2px 8px; border-radius: 8px; background: rgba(56, 189, 248, 0.16); border: 1px solid rgba(56, 189, 248, 0.45); color: #7DD3FC; font-size: 0.73rem; font-weight: 700; }}
    h1, h2, h3, h4 {{ color: {cfg['accent']} !important; text-align: center; font-weight: 700; }}
    .stButton>button {{ background-color: {cfg['card']} !important; color: {cfg['accent']} !important; border: 2px solid {cfg['accent']}88 !important; border-radius: 12px !important; font-weight: 700 !important; }}
    .stButton>button:hover {{ background-color: {cfg['accent']} !important; color: #FFFFFF !important; }}
    </style>
    {generar_html_animacion_fondo(cond_actual_real)}
""", unsafe_allow_html=True)


def renderizar_tarjeta_dia(row, simbolo_temp):
    lluvia_html = f"<div class='lluvia-badge'>🌧️ Prob. lluvia: {row['Lluvia']}%</div>" if int(row["Lluvia"]) > 0 else ""
    st.markdown(f"""
        <div class="tarjeta-clima-custom">
            <div class="fecha-tarjeta">{row['FechaTitulo']}</div>
            <div class="temp-tarjeta">{row['TMAX']}{simbolo_temp}</div>
            <div class="min-tarjeta">Mín: {row['TMIN']}{simbolo_temp}</div>
            <div class="extra-detalles">💧 Hum.: {row['Humedad']}% &nbsp;•&nbsp; ☀️ UV: {row['UV']}</div>
            {lluvia_html}
        </div>
        <div style='text-align: center;'><span style='background-color: {cfg['badge']}; color: {cfg['accent']}; padding: 3px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: bold;'>{row['Icono']} {row['Tipo']}</span></div>
        {row['GatitoHtml']}
    """, unsafe_allow_html=True)
    if st.button("📊 Ver detalle", key=f"btn_{row['ID']}", use_container_width=True):
        st.session_state.dia_seleccionado = int(row["ID"])
        st.session_state.vista_detalle = True
        st.session_state.vista_actualizaciones = False
        st.rerun()


if st.session_state.vista_actualizaciones:
    if st.button("⬅️ Volver al inicio"):
        st.session_state.vista_actualizaciones = False
        st.rerun()
    st.title(f"🆕 Actualizaciones • Versión {VERSION_APP}")
    st.markdown(f"<p style='text-align: center; color: {cfg['subtext']}; font-size: 1rem;'>Novedades desde la versión 2.1 hasta la {VERSION_APP}</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="tarjeta-clima-custom" style="text-align: left; padding: 20px;">
                <div class="fecha-tarjeta">🗓️ Pronóstico Semanal de 7 Días</div>
                <p style="font-size: 0.92rem; line-height: 1.5; margin-bottom: 14px;">Se amplió el tablero a <b>7 días</b> con vista horaria detallada, indicador de probabilidad de lluvia y gatito lluvioso solo desde <b>50%</b>.</p>
                <div class="fecha-tarjeta">🥶 Nuevo Clima 'Frío' y 'Noche fría' (≤12°C)</div>
                <p style="font-size: 0.92rem; line-height: 1.5; margin-bottom: 14px;">Se agregó el modo <b>Frío</b> y <b>Noche fría</b> cuando la temperatura es de <b>12°C o menor</b>, separado de <b>Nevado</b> y manteniendo el recuadro coherente con el fondo.</p>
                <div class="fecha-tarjeta">📍 Selector Rápido y Búsqueda</div>
                <p style="font-size: 0.92rem; line-height: 1.5; margin-bottom: 0;">Menú rápido para elegir <b>Saltillo</b>, <b>Arteaga</b> o buscar cualquier ciudad.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="tarjeta-clima-custom" style="text-align: left; padding: 20px;">
                <div class="fecha-tarjeta">📂 Lectura Universal de CSV Históricos</div>
                <p style="font-size: 0.92rem; line-height: 1.5; margin-bottom: 14px;"><code>datos_limpios_salt.csv</code> y <code>datos_limpios_arte.csv</code> se leen automáticamente detectando codificación, separador, formatos de fecha y columnas <code>FECHA</code>, <code>TMAX</code> y <code>TMIN</code>, con caché de 366 días.</p>
                <div class="fecha-tarjeta">💬 3 Frases Rotativas y Encabezado Limpio</div>
                <p style="font-size: 0.92rem; line-height: 1.5; margin-bottom: 0;">Cada clima rota entre <b>3 frases</b> y se retiró el texto auxiliar de API + Datos Históricos del título principal.</p>
            </div>
        """, unsafe_allow_html=True)

elif not st.session_state.vista_detalle:
    st.title(f"Clima Bonito • {municipio_limpio}, {estado_limpio}")
    sim_t = st.session_state.unidades_temp
    lluvia_txt = f" | 🌧️ Prob. lluvia: <b>{lluvia_actual_val}%</b>" if lluvia_actual_val > 0 else ""
    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; font-weight: 600;'>Ahora: <b style='color: {cfg['accent']};'>{temp_actual_val}{sim_t} • {cond_actual_real}</b> {icono_actual_real}{lluvia_txt}</p>", unsafe_allow_html=True)

    cols1 = st.columns(4)
    for idx in range(min(4, len(df_p))):
        with cols1[idx]:
            renderizar_tarjeta_dia(df_p.iloc[idx], sim_t)
    st.markdown("<br>", unsafe_allow_html=True)
    if len(df_p) > 4:
        cols2 = st.columns(3)
        for j, idx in enumerate(range(4, len(df_p))):
            with cols2[j]:
                renderizar_tarjeta_dia(df_p.iloc[idx], sim_t)

else:
    dia_info = df_p[df_p["ID"] == st.session_state.dia_seleccionado].iloc[0]
    sim_t = st.session_state.unidades_temp
    if st.button("⬅️ Volver al inicio"):
        st.session_state.vista_detalle = False
        st.rerun()

    st.title(f"Detalle del Clima • {dia_info['FechaTitulo']}")
    lluvia_det = f" | 🌧️ Prob. lluvia: <b>{dia_info['Lluvia']}%</b>" if int(dia_info["Lluvia"]) > 0 else ""
    st.markdown(f"<p style='text-align: center; color: {cfg['accent']}; font-size: 1.2rem; font-weight: 700;'>Máxima: <b>{dia_info['TMAX']}{sim_t}</b> | Mínima: <b>{dia_info['TMIN']}{sim_t}</b> {dia_info['Icono']}{lluvia_det}</p>", unsafe_allow_html=True)

    t_max_sel, t_min_sel = dia_info["TMAX"], dia_info["TMIN"]
    horas, temps, iconos_h, desc_h, lluvias_h = [], [], [], [], []
    if datos_api_raw and "hourly" in datos_api_raw and "weathercode" in datos_api_raw["hourly"]:
        try:
            d_idx = int(st.session_state.dia_seleccionado)
            i0, i1 = d_idx * 24 + 6, d_idx * 24 + 24
            t_api = datos_api_raw["hourly"]["temperature_2m"][i0:i1]
            c_api = datos_api_raw["hourly"]["weathercode"][i0:i1]
            p_api = datos_api_raw["hourly"].get("precipitation_probability", [])[i0:i1]
            for k_h, h in enumerate(range(6, 24)):
                t_raw_c = float(t_api[k_h]) if k_h < len(t_api) else float(t_min_sel)
                c_val = c_api[k_h] if k_h < len(c_api) else 0
                p_val = int(p_api[k_h]) if k_h < len(p_api) and p_api[k_h] is not None else 0
                cond_hr, icon_hr = interpretar_weathercode(c_val, h, 0, temp_ref_c=t_raw_c)
                horas.append(f"{h:02d}:00")
                temps.append(convertir_temp(t_raw_c))
                iconos_h.append(icon_hr)
                desc_h.append(f"{icon_hr} {cond_hr}")
                lluvias_h.append(p_val)
        except Exception:
            horas, temps, iconos_h, desc_h, lluvias_h = [], [], [], [], []

    if not horas:
        for h in range(6, 24):
            factor = abs(math.sin(math.pi * (h - 6) / 16))
            horas.append(f"{h:02d}:00")
            temps.append(round(t_min_sel + factor * (t_max_sel - t_min_sel), 1))
            iconos_h.append(dia_info["Icono"])
            desc_h.append(f"{dia_info['Icono']} {dia_info['Tipo']}")
            lluvias_h.append(int(dia_info["Lluvia"]))

    df_horas = pd.DataFrame({"Hora": horas, f"Temperatura ({sim_t})": temps, "Icono": iconos_h, "Condicion": desc_h, "ProbLluvia": lluvias_h})
    fig = px.area(df_horas, x="Hora", y=f"Temperatura ({sim_t})", markers=True, line_shape="spline")
    fig.update_traces(line=dict(color=cfg["accent"], width=3), marker=dict(size=7, color=cfg["accent"]), fill="tozeroy", fillcolor=f"{cfg['accent']}33")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Quicksand", color=cfg["text"], size=12), height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    cards_html = [
        f"""<div style="background: {cfg['card']}; border: 1.5px solid {cfg['accent']}44; border-radius: 14px; padding: 10px 6px; text-align: center; min-height: 98px; display: flex; flex-direction: column; justify-content: center;">
            <div style="color: {cfg['accent']}; font-size: 0.8rem; font-weight: 700; margin-bottom: 2px;">{rh['Hora']}</div>
            <div style="color: {cfg['val_color']}; font-size: 1.1rem; font-weight: 700; margin-bottom: 2px;">{rh[f'Temperatura ({sim_t})']}{sim_t}</div>
            <div style="font-size: 0.65rem; color: {cfg['subtext']};">{rh['Condicion']}</div>
            {f"<div style='font-size: 0.65rem; color: #7DD3FC; font-weight: 700; margin-top: 2px;'>🌧️ {int(rh['ProbLluvia'])}%</div>" if int(rh['ProbLluvia']) > 0 else ""}
        </div>"""
        for _, rh in df_horas.iterrows()
    ]
    components.html(f"""
    <div style="font-family: 'Quicksand', sans-serif;">
        <style>
            .grid-horas-nativa {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 15px; }}
            @media (max-width: 1024px) {{ .grid-horas-nativa {{ grid-template-columns: repeat(3, 1fr); }} }}
            @media (max-width: 580px) {{ .grid-horas-nativa {{ grid-template-columns: repeat(2, 1fr); }} }}
        </style>
        <div class="grid-horas-nativa">{"".join(cards_html)}</div>
    </div>
    """, height=350, scrolling=True)
