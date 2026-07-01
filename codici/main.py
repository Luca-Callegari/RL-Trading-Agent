import yfinance as yf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque

# Importiamo la classe TradingAgent dal file agent.py
from agent import TradingAgent

choose = int(input("QUALE ALGORITMO SI VUOLE UTILIZZARE?\n1. SARSA\n2. Q-LEARNING\n3. DOUBLE-QLEARNING\n"))

if (choose == 1):
        # Importiamo la classe SARSAAgent dal file sarsa.py
        from SARSA import SARSAAgent
elif (choose == 2):
        # Importiamo la classe Q_LEARNINGagent dal file Q_LEARNING.py
        from Q_LEARNING import Q_LEARNINGagent
elif (choose == 3):
        # Importiamo la classe DQ_LEARNINGagent dal file DOUBLE_QLEARNING.py
        from DOUBLE_QLEARNING import DQ_LEARNINGagent
else:
        print("Comando non riconosciuto\n")
        sys.exit(1)

#----FASE DI PREPROCESSING----

tickers = ["BTC-USD", "ETH-USD", "GLD", "AAPL", "MSFT"]
start_date = "2020-01-01"
end_date = "2024-01-01"

# Scarichiamo direttamente la colonna 'Close' per tutti i ticker
data = yf.download(tickers, start = start_date, end = end_date)['Close']

# Usiamo ffill() per dire: "Se oggi la borsa è chiusa, il prezzo è lo stesso di ieri"
price_df = data.ffill().dropna()

# Resettiamo l'indice per avere la colonna 'Date'
price_df.reset_index(inplace = True)

# Salvataggio dei dati di mercato in un file CSV
price_df.to_csv("market_prices_rl.csv", index = False)

print("CSV con PREZZI creato con successo!")
print(price_df.head())

price_scale = price_df[tickers].max().values
cash_scale = 100000.0

#----FASE DI SETTAGGIO AMBIENTE/AGENTE----

# Inizializziamo l'ambiente 
env = TradingAgent(tickers)

# Inizializziamo l'agente (la funzione qualità viene inizializzata automaticamente)
if (choose == 1):
        brain = SARSAAgent(tickers, price_scale, cash_scale)
elif (choose == 2):
        brain = Q_LEARNINGagent(tickers, price_scale, cash_scale)
else:
        brain = DQ_LEARNINGagent(tickers, price_scale, cash_scale)

price_df = pd.read_csv("market_prices_rl.csv")

print(f"Capitale Iniziale: {env.cash_disponibile} $")
print("-" * 30)

#----FASE DI INIZIALIZZAZIONE----

# Numero di episodi
num_episodes = 5000
# Variabili di controllo
epsilon = 1 # Fattore di scelta greedy dell'azione
epsilon_decay = 3 # Fattore di esplorazione
alpha = 0.01  # Fattore di apprendimento
gamma = 0.99   # Fattore di sconto

# Vettore del tempo
x = []
# Vettore del valore portafoglio
y = []
# Vettore di norme
norm = []
# Vettore degli errori relativi
rel_errors = []

#----FASE DI APPRENDIMENTO----

for e in range(num_episodes):

    # Inizializziamo l'ambiente 
    env = TradingAgent(tickers)

    # Reset della cronologia per il nuovo episodio
    brain.history = [deque(maxlen = 5) for _ in range(brain.n_assets)]

    # Reset memoria prezzi per il nuovo episodio
    brain.prezzi_precedenti = None

    # Inizializziamo lo stato nel primo giorno
    curr_state = env.get_state(0, price_df)

    # Discretizziamo lo stato in un vettore 1xn_asset dove in ogni cella c'è un numero intero tra 0 e 7 (codifica di curr_state)
    s = brain.get_discrete_state_vector(curr_state)

    # Accumulatore batch
    batch_exp = []

    # Salvataggio Q-table per l'episodio precedente
    if choose == 3:
        q_old = brain.Q1.copy() + brain.Q2.copy()
    else:
        q_old = brain.Q.copy()

    if (choose == 1):
        # Scelta azione greedy per ogni asset, dato lo stato corrente. a = vettore 1x5
        a = brain.greedy_action(s, epsilon)

    # Ciclo per ogni riga del file CSV
    for t in range(len(price_df) - 1):

        prezzi_attuali = price_df.iloc[t][tickers].values
        prezzi_domani = price_df.iloc[t + 1][tickers].values

        # Calcolo del valore portafoglio attuale
        if e == num_episodes - 1:
            valore_attuale = env.cash_disponibile + np.sum(env.quantita_possedute * prezzi_attuali)
            y.append(valore_attuale)
            x.append(price_df.iloc[t]['Date'])

        if (choose == 2 or choose == 3):
            # Scelta azione greedy per ogni asset, dato lo stato corrente. a = vettore 1x5
            a = brain.greedy_action(s, epsilon)

        # Giornata di trading
        next_state, rewards = env.step(a, prezzi_attuali, prezzi_domani, brain.history)

        # Discretizziamo lo stato in un vettore 1xn_asset dove in ogni cella c'è un numero intero tra 0 e 7 (codifica di curr_state)
        sp = brain.get_discrete_state_vector(next_state)

        if (choose == 1):
            # Scelta azione greedy per ogni asset, dato lo stato successivo. a = vettore 1x5
            ap = brain.greedy_action(sp, epsilon)
            batch_exp.append((s, a, rewards, sp, ap))
            a = ap
        else:
            batch_exp.append((s, a, rewards, sp, None))

        s = sp

    # Aggiornamento BATCH a fine episodio
    brain.batch_update(batch_exp, alpha, gamma)

    if choose == 3:
        q_new = brain.Q1 + brain.Q2
    else:
        q_new = brain.Q
    
    # Calcolo della norma della differenza
    diff_norm = np.linalg.norm(q_new - q_old)
    norm.append(diff_norm)

    # Calcolo dell'errore relativo
    norm_q_old = np.linalg.norm(q_old)
    error = diff_norm / (norm_q_old + 1e-8)
    rel_errors.append(error)

    if (e+1) % 1000 == 0:
        # Stampa di controllo a fine episodio
        #valore_finale = env.cash_disponibile + np.sum(env.quantita_possedute * price_df.iloc[-1][tickers].values)
        print(f"Episodio {e+1}/{num_episodes} completato.")

    # Alla fine dell'episodio, riduci epsilon
    epsilon = max(0.15, np.exp(-epsilon_decay * e / num_episodes))

# ----GRAFICI----

# Grafico 1: Andamento Portafoglio
plt.figure(figsize = (12, 10))

plt.subplot(2, 1, 1)
plt.plot(x, y, color = 'green', linewidth = 2)
plt.xticks(rotation = 45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) 
plt.title(f"Andamento Portafoglio")
plt.ylabel("Valore Totale ($)")
plt.grid(True, linestyle = '--', alpha = 0.7)

# Grafico 2: Convergenza
""" plt.subplot(3, 1, 2)
plt.plot(range(num_episodes), norm, color = 'blue')
plt.title("Convergenza della Funzione Valore")
plt.xlabel("Episodio")
plt.ylabel("Norma della differenza (||Q_new - Q_old||)")
plt.yscale('log')
plt.grid(True, linestyle = '--', alpha = 0.7) """

# Grafico 3: Errore relativo
plt.subplot(2, 1, 2)
plt.plot(range(num_episodes), rel_errors, color = 'red')
plt.title("Errore Relativo della Funzione Valore")
plt.xlabel("Episodio")
plt.ylabel("Errore Relativo")
plt.yscale('log')
plt.grid(True, linestyle = '--', alpha = 0.7)

plt.tight_layout()
plt.subplots_adjust(hspace = 0.4)
plt.show()