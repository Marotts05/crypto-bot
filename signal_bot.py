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

# --- Rischio: valori di DEFAULT per chi si registra senza specificare un saldo ---
DEFAULT_ACCOUNT_BALANCE = 50.0
DEFAULT_RISK_PER_TRADE_PCT = 1.0

# --- Telegram ---
# Su GitHub Actions questi arrivano dai "repository secrets" (vedi guida).
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "INSERISCI_IL_TUO_TOKEN")
ADMIN_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")   # solo per migrare le vecchie impostazioni, vedi migrate_legacy_config()
COMMAND_CHECK_TIMEOUT = 5    # secondi di attesa quando controlliamo se sono arrivati comandi

# --- File di stato (salvati nella stessa cartella dello script, il workflow li ricommitta ad ogni run) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "last_signals.json")   # per evitare di mandare lo stesso segnale più volte
USERS_FILE = os.path.join(BASE_DIR, "users.json")          # {chat_id: {account_balance, risk_per_trade_pct}} per OGNI persona registrata
LEGACY_CONFIG_FILE = os.path.join(BASE_DIR, "bot_config.json")  # file della vecchia versione mono-utente, solo per migrazione


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

def detect_signal(df: pd.DataFrame) -> dict | None:
    """Guarda l'ultima candela chiusa e decide se c'è un segnale. Uguale per tutti
    gli utenti: entry/stop/take dipendono solo dal mercato, non dal capitale di nessuno."""
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

    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "sl_distance": sl_distance,
        "rsi": last["rsi"],
        "candle_time": df.index[-2],
    }


def compute_position(signal: dict, account_balance: float, risk_per_trade_pct: float) -> dict:
    """Calcola quanto comprare/vendere PER UNA SPECIFICA PERSONA, dato il suo saldo e rischio."""
    risk_amount = account_balance * (risk_per_trade_pct / 100)
    position_size = risk_amount / signal["sl_distance"]
    position_value = position_size * signal["entry"]
    return {
        "position_size": position_size,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "risk_pct": risk_per_trade_pct,
    }


# ============ NOTIFICHE TELEGRAM ============

def send_telegram_message(chat_id: str, text: str):
    if "INSERISCI" in TELEGRAM_BOT_TOKEN:
        print(f"[ATTENZIONE] Telegram non configurato. Messaggio per {chat_id} che sarebbe stato inviato:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Errore invio Telegram a {chat_id}: {e}")


def format_signal_message(symbol: str, signal: dict, position: dict) -> str:
    direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    return (
        f"{direction_emoji} <b>{signal['direction']} {symbol}</b>\n"
        f"Candela: {signal['candle_time']}\n\n"
        f"Entry: {signal['entry']:.4f}\n"
        f"Stop Loss: {signal['stop_loss']:.4f}\n"
        f"Take Profit: {signal['take_profit']:.4f}\n"
        f"RSI: {signal['rsi']:.1f}\n\n"
        f"💰 Size posizione: {position['position_size']:.6f} (~{position['position_value']:.2f} USDT)\n"
        f"⚠️ Rischio su questo trade: {position['risk_amount']:.2f} USDT ({position['risk_pct']}% capitale)\n"
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


# ============ UTENTI REGISTRATI (ognuno col proprio saldo/rischio) ============

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def migrate_legacy_config(users: dict) -> dict:
    """Una tantum: se esiste il vecchio file mono-utente e l'admin non è ancora
    tra gli utenti registrati, lo migra così non perde saldo/rischio già impostati."""
    if ADMIN_CHAT_ID and ADMIN_CHAT_ID not in users and os.path.exists(LEGACY_CONFIG_FILE):
        try:
            with open(LEGACY_CONFIG_FILE, "r") as f:
                old = json.load(f)
            users[ADMIN_CHAT_ID] = {
                "account_balance": old.get("account_balance", DEFAULT_ACCOUNT_BALANCE),
                "risk_per_trade_pct": old.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT),
            }
            save_users(users)
            print(f"Migrato utente admin {ADMIN_CHAT_ID} dal vecchio bot_config.json")
        except Exception as e:
            print(f"Migrazione fallita ({e}), ignoro.")
    return users


def handle_command(text: str, chat_id: str, users: dict) -> dict:
    parts = text.strip().split()
    if not parts:
        return users
    command = parts[0].lower()
    registrato = chat_id in users

    if command == "/registrati":
        if registrato:
            send_telegram_message(chat_id, "Sei già registrato! Scrivi /stato per vedere le tue impostazioni.")
        else:
            saldo_iniziale = DEFAULT_ACCOUNT_BALANCE
            if len(parts) == 2:
                try:
                    saldo_iniziale = float(parts[1].replace(",", "."))
                except ValueError:
                    send_telegram_message(chat_id, "⚠️ Formato non valido. Scrivi ad esempio: /registrati 100")
                    return users
            users[chat_id] = {"account_balance": saldo_iniziale, "risk_per_trade_pct": DEFAULT_RISK_PER_TRADE_PCT}
            save_users(users)
            send_telegram_message(
                chat_id,
                f"✅ Registrato! Saldo iniziale: {saldo_iniziale:.2f}, rischio: {DEFAULT_RISK_PER_TRADE_PCT}% a trade.\n"
                f"Da ora ricevi i segnali. Scrivi /help per la lista comandi."
            )

    elif command in ("/esci", "/cancella", "/disiscriviti"):
        if registrato:
            del users[chat_id]
            save_users(users)
            send_telegram_message(chat_id, "Fatto, non ricevi più segnali. Scrivi /registrati per tornare quando vuoi.")
        else:
            send_telegram_message(chat_id, "Non risulti registrato.")

    elif command == "/saldo":
        if not registrato:
            send_telegram_message(chat_id, "Non sei ancora registrato. Scrivi /registrati 100 per iniziare.")
        elif len(parts) == 2:
            try:
                nuovo_saldo = float(parts[1].replace(",", "."))
                users[chat_id]["account_balance"] = nuovo_saldo
                save_users(users)
                send_telegram_message(chat_id, f"✅ Saldo aggiornato a {nuovo_saldo:.2f}. Verrà usato dal prossimo controllo.")
            except ValueError:
                send_telegram_message(chat_id, "⚠️ Formato non valido. Scrivi ad esempio: /saldo 200")
        else:
            send_telegram_message(chat_id, "⚠️ Manca l'importo. Scrivi ad esempio: /saldo 200")

    elif command == "/rischio":
        if not registrato:
            send_telegram_message(chat_id, "Non sei ancora registrato. Scrivi /registrati 100 per iniziare.")
        elif len(parts) == 2:
            try:
                nuovo_rischio = float(parts[1].replace(",", "."))
                if 0 < nuovo_rischio <= 10:
                    users[chat_id]["risk_per_trade_pct"] = nuovo_rischio
                    save_users(users)
                    send_telegram_message(chat_id, f"✅ Rischio per trade aggiornato a {nuovo_rischio:.1f}%.")
                else:
                    send_telegram_message(chat_id, "⚠️ Usa un valore tra 0 e 10, es. /rischio 1.5")
            except ValueError:
                send_telegram_message(chat_id, "⚠️ Formato non valido. Scrivi ad esempio: /rischio 1.5")
        else:
            send_telegram_message(chat_id, "⚠️ Manca il valore. Scrivi ad esempio: /rischio 1.5")

    elif command == "/stato":
        if not registrato:
            send_telegram_message(chat_id, "Non sei ancora registrato. Scrivi /registrati 100 per iniziare.")
        else:
            u = users[chat_id]
            coppie = f"automatica (top {TOP_N_SYMBOLS})" if AUTO_SELECT_SYMBOLS else ", ".join(SYMBOLS)
            send_telegram_message(
                chat_id,
                f"📊 <b>Le tue impostazioni</b>\n"
                f"Saldo: {u['account_balance']:.2f}\n"
                f"Rischio per trade: {u['risk_per_trade_pct']:.1f}%\n"
                f"Coppie monitorate: {coppie}\n"
                f"Timeframe: {TIMEFRAME}"
            )

    elif command in ("/help", "/aiuto", "/start"):
        send_telegram_message(
            chat_id,
            "🤖 <b>Comandi disponibili</b>\n"
            "/registrati 100 → iscriviti e inizia a ricevere segnali (100 = tuo saldo)\n"
            "/saldo 200 → cambia il capitale considerato\n"
            "/rischio 1.5 → cambia il % di rischio per trade (max 10)\n"
            "/stato → mostra le tue impostazioni\n"
            "/esci → smetti di ricevere segnali"
        )

    else:
        send_telegram_message(chat_id, "Non riconosco questo comando. Scrivi /help per la lista.")

    return users


def check_telegram_commands(update_offset, users: dict):
    """Controlla se sono arrivati messaggi su Telegram, da chiunque, e li applica."""
    if "INSERISCI" in TELEGRAM_BOT_TOKEN:
        return update_offset, users

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": COMMAND_CHECK_TIMEOUT}
    if update_offset is not None:
        params["offset"] = update_offset

    try:
        resp = requests.get(url, params=params, timeout=COMMAND_CHECK_TIMEOUT + 10)
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"Errore nel controllo comandi Telegram: {e}")
        return update_offset, users

    for update in updates:
        update_offset = update["update_id"] + 1
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")

        if chat_id and text:
            users = handle_command(text, chat_id, users)

    return update_offset, users


# ============ ESECUZIONE (un giro solo: comandi + scansione + salvataggio) ============

def main():
    exchange = getattr(ccxt, EXCHANGE_ID)()
    state = load_state()
    users = load_users()
    users = migrate_legacy_config(users)

    # 1) Applica eventuali comandi Telegram arrivati dall'ultima esecuzione (da chiunque abbia scritto)
    _, users = check_telegram_commands(None, users)

    # 2) Decidi quali coppie guardare in questo giro
    active_symbols = get_top_symbols(exchange) if AUTO_SELECT_SYMBOLS else SYMBOLS

    modalita = f"auto (top {TOP_N_SYMBOLS} per volume)" if AUTO_SELECT_SYMBOLS else "manuale"
    print(f"[{datetime.now()}] Bot avviato [{modalita}]. Utenti registrati: {len(users)}.")
    print(f"Monitoro {len(active_symbols)} coppie su timeframe {TIMEFRAME}: {active_symbols}")

    if not users:
        print("Nessun utente registrato: nessuno riceverà segnali finché qualcuno non scrive /registrati.")

    # 3) Scansiona il mercato una volta (uguale per tutti: entry/stop/take non dipendono dall'utente)
    for symbol in active_symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLES_LIMIT)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            df = add_indicators(df)
            signal = detect_signal(df)

            if signal:
                candle_key = f"{symbol}_{signal['candle_time']}"
                if state.get(symbol) != candle_key:
                    # 4) Segnale nuovo: manda a OGNI utente registrato la SUA size personale
                    for chat_id, u in users.items():
                        position = compute_position(signal, u["account_balance"], u["risk_per_trade_pct"])
                        msg = format_signal_message(symbol, signal, position)
                        send_telegram_message(chat_id, msg)
                    print(f"{symbol}: segnale {signal['direction']} mandato a {len(users)} utenti.")
                    state[symbol] = candle_key
            else:
                print(f"{symbol}: nessun segnale.")

        except Exception as e:
            print(f"Errore su {symbol}: {e}")

    # 5) Salva lo stato aggiornato (il workflow lo ricommitta nel repository)
    save_state(state)
    print("Giro completato.")


if __name__ == "__main__":
    main()
