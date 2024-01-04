import copy
import pickle
import os
import random
import NN

import numpy as np
import cupy as cp


class DDQN:

    def __init__(self, sit, koeficientUceni, epsilon, gamma, nKroku, velikostBufferu, mozneAkce,
                 funkceChyby=NN.ChybovkaE):
        self.sitOnline = NN.NN(sit, koeficientUceni)
        self.sitCilova = copy.deepcopy(self.sitOnline)
        self.epsion = epsilon
        self.gamma = gamma
        self.nKroku = nKroku
        self.funkceChyby = funkceChyby
        self.zbyvaKroku = nKroku
        self.mozneAkce = mozneAkce
        self.velikostBufferu = velikostBufferu
        self.buffer = []

    def ziskejAkci(self, stav):  # mozneAkce -> pocet moznych akci
        if np.random.random_sample() < self.epsion:
            akce = np.random.randint(self.mozneAkce)
        else:
            akce = np.argmax(cp.asnumpy(self.sitOnline.forward(stav)))
        return akce

    def pridejDoBufferu(self, stAkOdNsH):  # [stav, akce, odmena, novyStav, hotovo]
        if len(self.buffer) > self.velikostBufferu:
            del self.buffer[0]
        self.buffer.append(stAkOdNsH)

    def vemZBufferu(self):
        return random.choice(self.buffer)

    def aktualizuj(self, stav, akce, odmena, novyStav, hotovo):
        if not hotovo:  # nebot to uz nikam nevede
            cil = odmena + np.multiply(self.gamma, cp.asnumpy(
                self.sitCilova.forward(novyStav)[cp.argmax(self.sitOnline.forward(novyStav))]))
        else:
            cil = odmena
        momentalni = cp.asnumpy(self.sitOnline.forward(stav))[akce]
        tempChyba = self.funkceChyby.deriv(cil, momentalni)
        if self.zbyvaKroku == 0:
            self.sitCilova = copy.deepcopy(self.sitOnline)
            self.zbyvaKroku = self.nKroku
        chyba = np.zeros(self.mozneAkce)
        chyba[akce] = tempChyba
        self.sitOnline.back(chyba)
        self.zbyvaKroku -= 1

    def uloz(self, soubor="ddqn.bin"):
        with open(soubor, "wb") as temp:
            pickle.dump(self, temp)

    @staticmethod
    def nacti(soubor="ddqn.bin"):
        if os.path.isfile(soubor):
            with open(soubor, "rb") as temp:
                return pickle.load(temp)
        else:
            raise Exception(f"Soubor '{soubor}' neexistuje")
