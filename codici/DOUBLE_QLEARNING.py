import numpy as np
import random
from tilecoding import IHT,tiles
from collections import deque

class DQ_LEARNINGagent:

    def __init__(self, tickers, price_scale, cash_scale):

        self.tickers = tickers
        self.n_assets = len(tickers)
        self.price_scale = price_scale
        self.cash_scale = cash_scale

        # Inizializzazione dell'IHT
        self.iht_size = 65536
        self.iht = IHT(self.iht_size)

        # Verifica importazione
        print("DOUBLE-QLEARNING importato con successo!\n")
        
        # Q-Table aggiornata a 32 stati (righe) e 3 azioni (colonne)
        self.Q1 = np.zeros((self.n_assets, self.iht_size, 3))
        self.Q2 = np.zeros((self.n_assets, self.iht_size, 3))
        
        # History per il trend a 5 giorni
        self.history = [deque(maxlen=5) for _ in range(self.n_assets)]
    
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
                q_values = [np.sum(self.Q1[i, current_state_indices, a] + self.Q2[i, current_state_indices, a]) for a in range(3)]
                indice_azione = np.argmax(q_values)
            
            azioni_scelte.append(mappa_azioni[indice_azione])
            
        return np.array(azioni_scelte)
    
    def batch_update(self, experiences, alpha, gamma):
       
        # Aggiornamento batch per DOUBLE-QLEARNING:
        # Accumula i delta per Q1 e Q2 separatamente e aggiorna a fine episodio.

        inv_mappa = {-1: 0, 0: 1, 1: 2}
        
        # Inizializziamo due accumulatori per i delta, uno per Q1 e uno per Q2
        batch_delta_Q1 = np.zeros_like(self.Q1)
        batch_delta_Q2 = np.zeros_like(self.Q2)

        for s, a, r, s_next, _ in experiences:

            # Per ogni transizione, scegliamo casualmente quale tabella "colpire"
            update_q1 = np.random.random() <= 0.5
            
            for i in range(self.n_assets):

                idx_a = inv_mappa[a[i]]

                if update_q1:
                    # Usiamo Q1 per scegliere l'azione e Q2 per valutarla
                    current_val = np.sum(self.Q1[i, s, idx_a])
                    
                    # Azione migliore in s_next secondo Q1
                    best_next_a = np.argmax([np.sum(self.Q1[i, s_next, act]) for act in range(3)])
                    # Valore di quell'azione stimato da Q2
                    next_val_eval = np.sum(self.Q2[i, s_next, best_next_a])
                    
                    target = r + (gamma * next_val_eval)
                    delta = (alpha / 32) * (target - current_val)
                    
                    # Accumuliamo nel batch di Q1
                    batch_delta_Q1[i, s, idx_a] += delta
                else:
                    # Usiamo Q2 per scegliere l'azione e Q1 per valutarla
                    current_val = np.sum(self.Q2[i, s, idx_a])
                    
                    # Azione migliore in s_next secondo Q2
                    best_next_a = np.argmax([np.sum(self.Q2[i, s_next, act]) for act in range(3)])
                    # Valore di quell'azione stimato da Q1
                    next_val_eval = np.sum(self.Q1[i, s_next, best_next_a])
                    
                    target = r + (gamma * next_val_eval)
                    delta = (alpha / 32) * (target - current_val)
                    
                    # Accumuliamo nel batch di Q2
                    batch_delta_Q2[i, s, idx_a] += delta

        # Applichiamo gli aggiornamenti accumulati a fine episodio
        self.Q1 += batch_delta_Q1
        self.Q2 += batch_delta_Q2