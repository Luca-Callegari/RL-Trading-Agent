# RL-Trading-Agent 🤖📈

Questo progetto nasce con l'idea di mettere alla prova il **Reinforcement Learning (RL)** all'interno di uno dei contesti più caotici e imprevedibili in assoluto: il trading online. L'obiettivo è addestrare un agente intelligente capace di muoversi autonomamente in un mercato azionario simulato, imparando quando comprare, vendere o restare a guardare per massimizzare il profitto nel lungo termine.

---

## 1. Come abbiamo modellato l'ambiente di trading

Per far sì che l'agente potesse imparare qualcosa, abbiamo dovuto definire le regole del gioco (l'ambiente) e il modo in cui percepisce il mondo circostante.

### Discretizzazione dello stato: Tile Coding
Le variabili di un mercato finanziario (prezzi, variazioni, trend) si muovono in uno spazio continuo e potenzialmente infinito. Per permettere ad algoritmi tabulari di elaborare queste informazioni senza perdersi, abbiamo usato il **Tile Coding**. Questa tecnica riduce la complessità sovrapponendo più griglie sfasate (i *tilings*), permettendo all'agente di generalizzare quello che impara: se una scelta funziona in un determinato stato di mercato, l'agente saprà applicarla anche in condizioni molto simili.

### Azioni e Ricompense (Reward)
A ogni passo temporale, l'agente osserva il mercato e prende una decisione scegliendo tra tre azioni:
* **Buy**: Apre una posizione rialzista sull'asset.
* **Sell**: Chiude la posizione o vende l'asset.
* **Hold**: Rimane alla finestra, mantenendo la posizione attuale senza fare operazioni.

La funzione di *reward* ($R$) è il motore dell'apprendimento: restituisce un feedback matematico basato sul reale ritorno economico o sulla perdita generata dall'azione scelta. Se l'agente guadagna riceve un premio, se perde riceve una penalità.

---

## 2. I tre algoritmi a confronto (e le loro formule)

Per capire quale strategia di aggiornamento della funzione valore fosse la migliore, abbiamo implementato e testato tre approcci diversi basati su politiche di ottimizzazione di tipo *batch*:

### 📊 1. Agente SARSA (State-Action-Reward-State-Action)
SARSA è un approccio **on-policy**. Significa che l'agente è prudente: quando aggiorna la sua funzione di qualità $Q(S, A)$, valuta l'azione successiva seguendo la stessa identica politica (la politica $\epsilon$-greedy) che userà poi per muoversi davvero. Tiene conto del rischio reale del percorso che sta intraprendendo.

La formula di aggiornamento è:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma Q(S', A') - Q(S, A) \right)$$

### 📈 2. Agente Q-Learning
Il Q-Learning classico ragiona in modo **off-policy** ed è decisamente più aggressivo. Quando aggiorna la funzione valore guardando allo stato futuro $S'$, non gli importa della politica di esplorazione corrente: assume sempre che nel passo successivo farà la scelta migliore in assoluto (quella che massimizza la $Q$). Si basa sulle equazioni di ottimalità di Bellman.

La formula di aggiornamento è:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left( R + \gamma \max_{a} Q(S', a) - Q(S, A) \right)$$

*Nota di comportamento:* Nelle nostre simulazioni, questo ottimismo spinge l'agente a rischiare di più rispetto a SARSA, accettando operazioni più volatili pur di inseguire il massimo profitto teorico.

### 🛡️ 3. Agente Double Q-Learning
L'operatore di massimo ($\max$) del Q-Learning soffre di un difetto noto: la *sovrastima ottimistica* (o *positive bias*). In mercati rumorosi e stocastici, rischia di sovrastimare i guadagni reali. Per risolvere questo problema, abbiamo implementato il Double Q-Learning, che sdoppia la conoscenza dell'agente in due tabelle distinte ($Q_1$ e $Q_2$). Una tabella decide qual è l'azione migliore, l'altra ne valuta il valore effettivo.

L'aggiornamento avviene con probabilità del 50% su una delle due tabelle. Se aggiorniamo $Q_1$, la formula diventa:

$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha \left( R + \gamma Q_2\left(S', \arg\max_{a} Q_1(S', a)\right) - Q_1(S, A) \right)$$

Questo sdoppiamento corregge il bias, offrendo una stima dei profitti molto più realistica e una convergenza decisamente più stabile.

---

## 3. Cosa volevamo dimostrare

Con questo codice abbiamo voluto analizzare il bilanciamento ideale tra l'aggressività della ricerca del profitto e la gestione del rischio. Il confronto mette in luce come la scelta della struttura algoritmica (on-policy contro off-policy) e l'uso di tecniche per abbattere la sovrastima delle ricompense siano cruciali quando si applica l'intelligenza artificiale a dati complessi come le serie storiche finanziarie.

---
**Autori:** Mirko Damiano & Luca Callegari  
**Università:** Università degli Studi di Roma "Tor Vergata"  
**Corso:** Laurea Magistrale in Ingegneria dell'Automazione  
**Progetto per l'esame di:** Machine and Reinforcement Learning
