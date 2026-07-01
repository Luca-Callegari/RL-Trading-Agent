import numpy as np
import pandas as pd

class TradingAgent:

    def __init__(self, tickers):

        self.tickers = tickers
        self.n_assets = len(tickers)
        
        # Variabili originali
        self.cash_disponibile = 10000.0 # Dollari 
        self.quantita_possedute = np.zeros(self.n_assets) 
        
    def get_state(self, row_index, df):

        # Prezzi attuali
        prezzi_attuali = df.iloc[row_index][self.tickers].values
        
        # Valore attuale del portafoglio
        valore_portafoglio = self.cash_disponibile + np.sum(self.quantita_possedute * prezzi_attuali)
        
        # Stato = [Cash, Valore_Portfafoglio, Prezzi (x n_asset), Quantità (x n_asset)]
        state = np.concatenate([
            [self.cash_disponibile],
            [valore_portafoglio],
            prezzi_attuali,
            self.quantita_possedute
        ])
        return state
    
    # Funzione per calcolare l'aggiornamento dello stato
    def step(self, action_vettore, prezzi_attuali, prezzi_domani, history):

        # Valore totale PRIMA delle operazioni (per confronto perdita)
        valore_totale_attuale = self.cash_disponibile + np.sum(self.quantita_possedute * prezzi_attuali)

        # Parametri di gestione
        MAX_SELL_PERCENT = 0.3  
        TARGET_EXPOSURE = 0.15   
        
        for i, action in enumerate(action_vettore):

            prezzo_asset = prezzi_attuali[i]
            asset_value = self.quantita_possedute[i] * prezzo_asset
            esposizione = asset_value / valore_totale_attuale if valore_totale_attuale > 0 else 0

            if action == 1: # BUY
                perc_asset_buy = max(0.05, TARGET_EXPOSURE - esposizione)
                dollari_da_investire = self.cash_disponibile * perc_asset_buy 
                self.cash_disponibile -= dollari_da_investire
                self.quantita_possedute[i] += (dollari_da_investire / prezzo_asset)

            elif action == -1: # SELL
                perc_asset_sell = MAX_SELL_PERCENT * esposizione
                quantita_venduta = self.quantita_possedute[i] * perc_asset_sell
                self.cash_disponibile += (quantita_venduta * prezzo_asset)
                self.quantita_possedute[i] -= quantita_venduta

        # Calcolo nuovo valore totale con prezzi di domani
        nuovo_valore_portafoglio = self.cash_disponibile + np.sum(self.quantita_possedute * prezzi_domani)
        
        # ----REWARD BASE----

        rewards = (nuovo_valore_portafoglio - valore_totale_attuale) / valore_totale_attuale
        
        nuovo_stato = np.concatenate([
            [self.cash_disponibile],
            [nuovo_valore_portafoglio],
            prezzi_domani,
            self.quantita_possedute
        ])
    
        return nuovo_stato, rewards