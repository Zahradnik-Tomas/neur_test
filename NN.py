import pickle
import os

import numpy as np
from scipy.special import expit, softmax


class ChybovkaE:
    @staticmethod
    def funkce(spravne, vystupSite):
        return np.true_divide(np.sum(
            np.power(np.subtract(spravne, vystupSite),
                     2.0)), 2.0)

    @staticmethod
    def deriv(spravne, vystupSite):
        return np.subtract(spravne, vystupSite)


class Sigmoid:

    def forward(self, vstup):
        self.vystup = expit(vstup)
        return self.vystup

    def deriv(self):
        return np.multiply(self.vystup, np.subtract(1.0, self.vystup))

    def back(self, chyba, koeficientUceni):
        return np.multiply(self.deriv(), np.asarray(chyba))


class SoftMax:
    def forward(self, vstup):
        self.vystup = softmax(vstup)
        return self.vystup

    def deriv(self):
        return np.multiply(self.vystup, (np.subtract(np.identity(np.size(self.vystup)), self.vystup)).T)

    def back(self, chyba, koeficientUceni):
        return np.dot(self.deriv(), np.asarray(chyba))


class ReLu:
    def forward(self, vstup):
        self.vstup = np.asarray(vstup)
        return np.maximum(0.0, self.vstup)

    def deriv(self):
        return np.greater(self.vstup, 0.0)

    def back(self, chyba, koeficientUceni):
        return np.multiply(self.deriv(), np.asarray(chyba))


class SS:  # jako Skalarni Soucin

    def __init__(self, pocetVstupu, pocetVystupu, rozklad=0.9, alfa_regul=0.01):  # pocetVystupu je pocet Neuronu
        self.maticeVah = np.subtract(np.random.rand(pocetVstupu + 1, pocetVystupu), 0.5)  # + 1 je bias
        self.rozklad = rozklad
        self.prum = np.asarray([0])
        self.l2_regul = alfa_regul / ((pocetVstupu + 1) * pocetVystupu * 2.0)

    def forward(self, vstup):
        self.vstup = np.append(vstup, 1.0)
        return np.dot(self.vstup.T, self.maticeVah)

    def back(self, chyba, koeficientUceni):
        np_chyba = np.asarray(chyba)
        chybaRaketak = np.dot(self.maticeVah[:-1], np_chyba)
        grad = np.multiply(np.expand_dims(self.vstup, axis=0).T,
                           np_chyba)  # 'nejde' transponovat vektor ... proto to musi jit na matici
        self.prum = np.add(np.multiply(self.rozklad, self.prum), np.multiply((1.0 - self.rozklad), np.power(grad, 2)))
        regul = np.multiply(2.0 * self.l2_regul, self.maticeVah)
        self.maticeVah += np.multiply(koeficientUceni, np.true_divide(grad,
                                                                      np.add(np.sqrt(self.prum),
                                                                             0.00000001)))
        self.maticeVah -= regul
        return chybaRaketak


class NN:
    def __init__(self, sit, koeficientUceni, funkceChyby=None):
        self.sit = sit
        self.koeficientUceni = koeficientUceni
        self.funkceChyby = funkceChyby

    def forward(self, vstup):
        self.vystup = np.asarray(vstup)
        for vrstva in self.sit:
            self.vystup = vrstva.forward(self.vystup)
        return self.vystup

    def back(self, pocatecniChyba):
        self.chyba = np.asarray(pocatecniChyba)
        for vrstva in reversed(self.sit):
            self.chyba = vrstva.back(self.chyba, self.koeficientUceni)

    def ziskejChybu(self, spravne):
        if self.funkceChyby is not None:
            return self.funkceChyby.deriv(spravne, self.vystup)

    def uloz(self, soubor="neuronka.bin"):
        with open(soubor, "wb") as temp:
            pickle.dump(self, temp)

    @staticmethod
    def nacti(soubor="neuronka.bin"):
        if os.path.isfile(soubor):
            with open(soubor, "rb") as temp:
                return pickle.load(temp)
        else:
            raise Exception(f"Soubor '{soubor}' neexistuje")
