import pickle
import os

import cupy as cp  # snad pujde videt rozdil ve vykonu
import cupyx.scipy.special
import numpy as np


class ChybovkaE:
    @staticmethod
    def funkce(spravne, vystupSite):
        return np.true_divide(np.sum(
            np.power(np.subtract(cp.asnumpy(spravne), cp.asnumpy(vystupSite)),
                     2.0)), 2.0)  # numpy misto cupy, protoze to nechci hazet: cpu -> gpu -> cpu

    @staticmethod
    def deriv(spravne, vystupSite):
        return np.subtract(cp.asnumpy(spravne), cp.asnumpy(vystupSite))


class Sigmoid:

    def forward(self, vstup):
        self.vystup = cupyx.scipy.special.expit(cp.asarray(vstup, dtype=cp.float32))
        return self.vystup

    def deriv(self):
        return cp.multiply(self.vystup, cp.subtract(1.0, self.vystup))

    def back(self, chyba, koeficientUceni):
        return cp.multiply(self.deriv(), cp.asarray(chyba, dtype=cp.float32))


class SoftMax:
    def forward(self, vstup):
        self.vystup = cupyx.scipy.special.softmax(cp.asarray(vstup, dtype=cp.float32))
        return self.vystup

    def deriv(self):
        return cp.multiply(self.vystup, (cp.subtract(cp.identity(cp.size(self.vystup)), self.vystup)).T)

    def back(self, chyba, koeficientUceni):
        return cp.dot(self.deriv(), cp.asarray(chyba, dtype=cp.float32))


class ReLu:
    def forward(self, vstup):
        self.vstup = cp.asarray(vstup)
        return cp.maximum(0.0, self.vstup)

    def deriv(self):
        return cp.greater(self.vstup, 0.0)

    def back(self, chyba, koeficientUceni):
        return cp.multiply(self.deriv(), cp.asarray(chyba, dtype=cp.float32))


class SS:  # jako Skalarni Soucin

    def __init__(self, pocetVstupu, pocetVystupu, rozklad=0.9, alfa_regul=0.01):  # pocetVystupu je pocet Neuronu
        self.maticeVah = cp.subtract(cp.random.rand(pocetVstupu + 1, pocetVystupu), 0.5)  # + 1 je bias
        self.rozklad = rozklad
        self.prum = cp.asarray([0])
        self.l2_regul = alfa_regul / ((pocetVstupu + 1) * pocetVystupu * 2.0)

    def forward(self, vstup):
        self.vstup = cp.append(vstup, 1.0)
        return cp.dot(self.vstup.T, self.maticeVah)

    def back(self, chyba, koeficientUceni):
        cp_chyba = cp.asarray(chyba, dtype=cp.float32)
        chybaRaketak = cp.dot(self.maticeVah[:-1], cp_chyba)
        grad = cp.multiply(cp.expand_dims(self.vstup, axis=0).T,
                           cp_chyba)  # 'nejde' transponovat vektor ... proto to musi jit na matici
        self.prum = cp.add(cp.multiply(self.rozklad, self.prum), cp.multiply((1.0 - self.rozklad), cp.power(grad, 2)))
        regul = cp.multiply(2.0 * self.l2_regul, self.maticeVah)
        self.maticeVah += cp.multiply(koeficientUceni, cp.true_divide(grad,
                                                                      cp.add(cp.sqrt(self.prum),
                                                                             0.00000001)))
        self.maticeVah -= regul
        return chybaRaketak


class NN:
    def __init__(self, sit, koeficientUceni, funkceChyby=None):
        self.sit = sit
        self.koeficientUceni = koeficientUceni
        self.funkceChyby = funkceChyby

    def forward(self, vstup):
        self.vystup = cp.asarray(vstup, dtype=cp.float32)
        for vrstva in self.sit:
            self.vystup = vrstva.forward(self.vystup)
        return self.vystup

    def back(self, pocatecniChyba):
        self.chyba = cp.asarray(pocatecniChyba, dtype=cp.float32)
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
