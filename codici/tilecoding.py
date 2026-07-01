"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
"""

basehash = hash # Utilizza la funzione hash predefinita di Python

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval # Dimensione massima della tabella                        
        self.overfullCount = 0 # Contatore per monitorare se superiamo il limite
        self.dictionary = {} # Dizionario che memorizza {coordinate: indice}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        # Controlla se la tabella è piena
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj] # Se le coordinate esistono già, restituisce l'indice
        elif readonly: return None # Se è in sola lettura e non esiste, non fare nulla
        size = self.size
        count = self.count()
        if count >= size: # Se la tabella è piena
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size # inizia a usare l'hash per forzare un indice (collisione)
        else:
            d[obj] = count # Altrimenti assegna un nuovo indice progressivo
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly) # Usa la classe IHT
    if type(m)==int: return basehash(tuple(coordinates)) % m # Hashing modulo m
    if m==None: return coordinates # Ritorna le coordinate grezze (debug)

from math import floor, log
from itertools import zip_longest

# Funzione che prende dei valori continui (floats) e li mappa in una lista di indici.

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    # Moltiplica i float per il numero di tiling per scalarli correttamente
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    # Ogni "tiling" è una griglia leggermente spostata rispetto alle altre
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling] # La prima coordinata è l'ID del tiling stesso
        b = tiling
        for q in qfloats:
            # Calcola la coordinata della cella per questa dimensione
            coords.append( (q + b) // numtilings )
            # Lo spostamento asimmetrico (b += tilingX2) garantisce una 
            # distribuzione uniforme dei tiling nello spazio.
            b += tilingX2
        coords.extend(ints) # Aggiunge eventuali variabili già intere (es. azioni)
        # Trasforma le coordinate finali in un indice unico
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

# Questa funzione permette ad alcune dimensioni di "riavvolgersi" (wrapping), utile per variabili circolari

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles