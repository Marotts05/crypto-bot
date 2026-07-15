"""
CRYPTO SIGNAL BOT
==================
Genera segnali di trading (LONG/SHORT) su coppie crypto usando:
- Trend: EMA veloce vs EMA lenta
- Timing d'ingresso: RSI
- Rischio: ATR per calcolare stop loss, take profit e size della posizione

IMPORTANTE: questo bot NON esegue ordini. Manda solo un messaggio Telegram
con il segnale e i livelli di rischio. Decidi tu se e come agire.

Pensato per girare come job "one-shot": si avvia, fa un controllo completo
(comandi Telegram in sospeso + scansione del mercato), salva lo stato su
file, ed esce. La ripetizione ogni tot minuti è gestita da fuori (es. un
workflow GitHub Actions con uno scheduler cron), non da un ciclo infinito
dentro allo script.

Prima di usarlo con soldi veri:
1. Fai un backtest (vedi backtest.py)
2. Prova in "carta" per settimane annotando i risultati a mano
3. Solo dopo, eventualmente, capitale reale piccolo
"""

import ccxt
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime

# ============ CONFIGURAZIONE ============
EXCHANGE_ID = "bitget"
TIMEFRAME = "4h"            # timeframe delle candele (es. "1h", "4h", "1d")
CANDLES_LIMIT = 300         # candele da scaricare ad ogni controllo

# --- Selezione automatica delle crypto da monitorare ---
AUTO_SELECT_SYMBOLS = True          # True = il bot sceglie da solo quali coppie guardare
TOP_N_SYMBOLS = 15                  # quante coppie monitorare in contemporanea
MIN_QUOTE_VOLUME_USDT = 5_000_000   # volume minimo nelle 24h per essere considerata (liquidità)
EXCLUDED_BASES = {"USDC", "USDT", "FDUSD", "TUSD", "DAI", "BUSD"}  # stablecoin: escluse, non hanno trend

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]   # lista di riserva, usata solo se AUTO_SELECT_SYMBOLS = False

# --- Parametri strategia ---
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 35           # sotto questo livello + trend rialzista = possibile LONG
RSI_OVERBOUGHT = 65         # sopra questo livello + trend ribassista = possibile SHORT
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5     # distanza stop loss = ATR * questo moltiplicatore
RISK_REWARD_RATIO = 2.0     # take profit = distanza stop loss * questo rapporto

# --- Gestione del rischio (valori di partenza: modificabili anche via Telegram, vedi sotto) ---
ACCOUNT_BALANCE = 50.0       # il tuo capitale in USDT — modificabile in chat con /saldo
RISK_PER_TRADE_PCT = 1.0     # % di capitale rischiato per trade — modificabile in chat con /rischio

# --- Telegram ---
# Su GitHub Actions questi arrivano dai "repository secrets" (vedi guida).
# In locale/PythonAnywhere, se non imposti le variabili d'ambiente, usa i valori di default qui sotto.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "INSERISCI_IL_TUO_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "INSERISCI_IL_TUO_CHAT_ID")
COMMAND_CHECK_TIMEOUT = 5    # secondi di attesa quando controlliamo se sono arrivati comandi

# --- File di stato (salvati nella stessa cartella dello script, il workflow li ricommitta ad ogni run) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "last_signals.json")   # per evitare di mandare lo stesso segnale più volte
CONFIG_FILE = os.path.join(BASE_DIR, "bot_config.json")    # saldo/rischio aggiornabili via Telegram, persistono tra un run e l'altro


# ============ INDICATORI (calcolati manualmente, no librerie extra) ============

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    return df


# ============ SELEZIONE AUTOMATICA DELLE COPPIE ============

def get_top_symbols(exchange, top_n=TOP_N_SYMBOLS, min_volume=MIN_QUOTE_VOLUME_USDT) -> list:
    """
    Guarda TUTTE le coppie */USDT disponibili sull'exchange e restituisce
    le top_n più scambiate nelle ultime 24h (più liquidità = spread migliori
    ed eseguibilità più affidabile). Esclude le stablecoin.
    """
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        print(f"Errore nel recupero della lista coppie ({e}). Uso la lista manuale di riserva.")
        return SYMBOLS

    candidates = []
    for symbol, ticker in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        base = symbol.split("/")[0]
        if base in EXCLUDED_BASES:
            continue

        volume_usdt = ticker.get("quoteVolume")
        if not volume_usdt:  # alcuni exchange non forniscono quoteVolume direttamente
            base_volume = ticker.get("baseVolume") or 0
            last_price = ticker.get("last") or 0
            volume_usdt = base_volume * last_price

        if volume_usdt and volume_usdt >= min_volume:
            candidates.append((symbol, volume_usdt))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in candidates[:top_n]]

    if not top_symbols:
        print("Nessuna coppia supera la soglia di volume impostata. Uso la lista manuale di riserva.")
        return SYMBOLS

    return top_symbols


# ============ LOGICA DEL SEGNALE ============

def detect_signal(df: pd.DataFrame, account_balance: float, risk_per_trade_pct: float) -> dict | None:
    """Guarda l'ultima candela chiusa e decide se c'è un segnale."""
    if len(df) < max(EMA_SLOW, ATR_PERIOD, RSI_PERIOD) + 5:
        return None  # non abbastanza dati

    last = df.iloc[-2]  # -2 = ultima candela CHIUSA (l'ultima riga -1 è ancora in corso)

    trend_up = last["ema_fast"] > last["ema_slow"]
    trend_down = last["ema_fast"] < last["ema_slow"]

    if trend_up and last["rsi"] < RSI_OVERSOLD:
        direction = "LONG"
    elif trend_down and last["rsi"] > RSI_OVERBOUGHT:
        direction = "SHORT"
    else:
        return None

    entry = last["close"]
    atr_val = last["atr"]
    if pd.isna(atr_val) or atr_val == 0:
        return None

    sl_distance = atr_val * ATR_SL_MULTIPLIER
    if direction == "LONG":
        stop_loss = entry - sl_distance
        take_profit = entry + sl_distance * RISK_REWARD_RATIO
    else:
        stop_loss = entry + sl_distance
        take_profit = entry - sl_distance * RISK_REWARD_RATIO

    risk_amount = account_balance * (risk_per_trade_pct / 100)
    position_size = risk_amount / sl_distance  # quantità in unità della crypto
    position_value = position_size * entry

    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "risk_pct": risk_per_trade_pct,
        "rsi": last["rsi"],
        "candle_time": df.index[-2],
    }


# ============ NOTIFICHE TELEGRAM ============

def send_telegram_message(text: str):
    if "INSERISCI" in TELEGRAM_BOT_TOKEN:
        print("[ATTENZIONE] Telegram non configurato. Messaggio che sarebbe stato inviato:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Errore invio Telegram: {e}")


def format_signal_message(symbol: str, signal: dict) -> str:
    direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    return (
        f"{direction_emoji} <b>{signal['direction']} {symbol}</b>\n"
        f"Candela: {signal['candle_time']}\n\n"
        f"Entry: {signal['entry']:.4f}\n"
        f"Stop Loss: {signal['stop_loss']:.4f}\n"
        f"Take Profit: {signal['take_profit']:.4f}\n"
        f"RSI: {signal['rsi']:.1f}\n\n"
        f"💰 Size posizione: {signal['position_size']:.6f} (~{signal['position_value']:.2f} USDT)\n"
        f"⚠️ Rischio su questo trade: {signal['risk_amount']:.2f} USDT ({signal['risk_pct']}% capitale)\n"
        f"R:R = 1:{RISK_REWARD_RATIO}"
    )


# ============ STATO (evita segnali duplicati) ============

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, default=str)


# ============ CONFIGURAZIONE MODIFICABILE VIA TELEGRAM ============

def load_config() -> dict:
    config = {"account_balance": ACCOUNT_BALANCE, "risk_per_trade_pct": RISK_PER_TRADE_PCT}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config.update(json.load(f))
        except Exception:
            pass
    return config


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def handle_command(text: str, config: dict) -> dict:
    parts = text.strip().split()
    if not parts:
        return config
    command = parts[0].lower()

    if command == "/saldo" and len(parts) == 2:
        try:
            nuovo_saldo = float(parts[1].replace(",", "."))
            config["account_balance"] = nuovo_saldo
            save_config(config)
            send_telegram_message(f"✅ Saldo aggiornato a {nuovo_saldo:.2f}. Verrà usato dal prossimo controllo.")
        except ValueError:
            send_telegram_message("⚠️ Formato non valido. Scrivi ad esempio: /saldo 200")

    elif command == "/rischio" and len(parts) == 2:
        try:
            nuovo_rischio = float(parts[1].replace(",", "."))
            if 0 < nuovo_rischio <= 10:
                config["risk_per_trade_pct"] = nuovo_rischio
                save_config(config)
                send_telegram_message(f"✅ Rischio per trade aggiornato a {nuovo_rischio:.1f}%.")
            else:
                send_telegram_message("⚠️ Usa un valore tra 0 e 10, es. /rischio 1.5")
        except ValueError:
            send_telegram_message("⚠️ Formato non valido. Scrivi ad esempio: /rischio 1.5")

    elif command == "/stato":
        coppie = f"automatica (top {TOP_N_SYMBOLS})" if AUTO_SELECT_SYMBOLS else ", ".join(SYMBOLS)
        send_telegram_message(
            f"📊 <b>Configurazione attuale</b>\n"
            f"Saldo: {config['account_balance']:.2f}\n"
            f"Rischio per trade: {config['risk_per_trade_pct']:.1f}%\n"
            f"Coppie: {coppie}\n"
            f"Timeframe: {TIMEFRAME}"
        )

    elif command in ("/help", "/aiuto", "/start"):
        send_telegram_message(
            "🤖 Comandi disponibili:\n"
            "/saldo 200 → cambia il capitale considerato\n"
            "/rischio 1.5 → cambia il % di rischio per trade (max 10)\n"
            "/stato → mostra la configurazione attuale"
        )

    return config


def check_telegram_commands(update_offset, config: dict):
    """Controlla se sono arrivati comandi su Telegram e aggiorna la configurazione di conseguenza."""
    if "INSERISCI" in TELEGRAM_BOT_TOKEN:
        return update_offset, config

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": COMMAND_CHECK_TIMEOUT}
    if update_offset is not None:
        params["offset"] = update_offset

    try:
        resp = requests.get(url, params=params, timeout=COMMAND_CHECK_TIMEOUT + 10)
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"Errore nel controllo comandi Telegram: {e}")
        return update_offset, config

    for update in updates:
        update_offset = update["update_id"] + 1
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")

        if chat_id == str(TELEGRAM_CHAT_ID) and text:
            config = handle_command(text, config)

    return update_offset, config


# ============ ESECUZIONE (un giro solo: comandi + scansione + salvataggio) ============

def main():
    exchange = getattr(ccxt, EXCHANGE_ID)()
    state = load_state()
    config = load_config()

    # 1) Applica eventuali comandi Telegram arrivati dall'ultima esecuzione
    _, config = check_telegram_commands(None, config)

    # 2) Decidi quali coppie guardare in questo giro
    active_symbols = get_top_symbols(exchange) if AUTO_SELECT_SYMBOLS else SYMBOLS

    modalita = f"auto (top {TOP_N_SYMBOLS} per volume)" if AUTO_SELECT_SYMBOLS else "manuale"
    print(f"[{datetime.now()}] Bot avviato [{modalita}]. Saldo: {config['account_balance']}, "
          f"rischio: {config['risk_per_trade_pct']}%.")
    print(f"Monitoro {len(active_symbols)} coppie su timeframe {TIMEFRAME}: {active_symbols}")

    # 3) Scansiona il mercato una volta
    for symbol in active_symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLES_LIMIT)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            df = add_indicators(df)
            signal = detect_signal(df, config["account_balance"], config["risk_per_trade_pct"])

            if signal:
                candle_key = f"{symbol}_{signal['candle_time']}"
                if state.get(symbol) != candle_key:
                    msg = format_signal_message(symbol, signal)
                    send_telegram_message(msg)
                    print(msg)
                    state[symbol] = candle_key
            else:
                print(f"{symbol}: nessun segnale.")

        except Exception as e:
            print(f"Errore su {symbol}: {e}")

    # 4) Salva lo stato aggiornato (il workflow lo ricommitta nel repository)
    save_state(state)
    print("Giro completato.")


if __name__ == "__main__":
    main()
