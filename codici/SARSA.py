import numpy as np
from tilecoding import IHT,tiles
from collections import deque

class SARSAAgent:

    def __init__(self, tickers, price_scale, cash_scale):

        self.tickers = tickers
        self.n_assets = len(tickers)
        self.price_scale = price_scale
        self.cash_scale = cash_scale

        # Inizializzazione dell'IHT
        self.iht_size = 65536
        self.iht = IHT(self.iht_size)

        # Verifica importazione
        print("SARSA importato con successo!\n")
        
        # Q-Table aggiornata a 32 stati (righe) e 3 azioni (colonne)
        self.Q = np.zeros((self.n_assets, self.iht_size, 3))
        
        # History per il trend a 5 giorni
        self.history = [deque(maxlen = 5) for _ in range(self.n_assets)]
    
    def get_discrete_state_vector(self, current_state):
        
        # Normalizzazione
        cash = current_state[0] / self.cash_scale
        portafoglio = current_state[1] / self.cash_scale
        prezzi_attuali = current_state[2 : 2 + self.n_assets] / self.price_scale
        quantita_possedute = current_state[2 + self.n_assets :] /10.0

        # Usiamo np.hstack per assicurarci che sia una lista piatta di soli numeri
        state_flat = np.hstack([
            cash, 
            portafoglio, 
            prezzi_attuali, 
            quantita_possedute
        ])

        indices = tiles(self.iht, 32, state_flat)

        return indices
    
    def greedy_action(self, current_state_indices, epsilon):

        azioni_scelte = []
        mappa_azioni = {0: -1, 1: 0, 2: 1}

        for i in range(self.n_assets):

            if np.random.random() < epsilon:
                indice_azione = np.random.randint(0, 3)
            else:
                q_values = [np.sum(self.Q[i, current_state_indices, a]) for a in range(3)]
                indice_azione = np.argmax(q_values)
            
            azioni_scelte.append(mappa_azioni[indice_azione])
            
        return np.array(azioni_scelte)
    
    def batch_update(self, experiences, alpha, gamma):

        # Aggiornamento batch per SARSA:
        # Calcola gli incrementi usando l'azione successiva scelta durante l'episodio.

        inv_mappa = {-1: 0, 0: 1, 1: 2}
        
        # Creiamo l'accumulatore temporaneo per i delta
        batch_delta = np.zeros_like(self.Q)

        # Iteriamo sulle esperienze 
        for s, a, r, sp, ap in experiences:

            for i in range(self.n_assets):

                idx_a = inv_mappa[a[i]]
                idx_ap = inv_mappa[ap[i]]

                # Calcolo curr_Q e next_Q
                curr_Q = np.sum(self.Q[i, s, idx_a])
                next_Q = np.sum(self.Q[i, sp, idx_ap])

                # Formula SARSA
                target = r + (gamma * next_Q)
                
                # Calcolo del delta
                delta = (alpha / 32) * (target - curr_Q)

                # Accumulo nel batch
                batch_delta[i, s, idx_a] += delta

        # Aggiornamento finale della Q-table
        self.Q += batch_delta