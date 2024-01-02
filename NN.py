import pickle
import os

import numpy as np  # nesnasim ho


# dalsi zbytecny komentar

class ChybovkaE:
    @staticmethod
    def funkce(spravne, vystupSite):
        return np.sum(np.power((np.array(spravne) - np.array(vystupSite)), 2)) / 2

    @staticmethod
    def deriv(spravne, vystupSite):
        return np.array(spravne) - np.array(vystupSite)


class Sigmoid:

    def forward(self, vstup):
        self.vstup = vstup
        vystup = []
        for cislo in vstup:
            if cislo >= 0:
                vystup.append(1 / (1 + np.exp(-cislo)))
            else:
                temp = np.exp(cislo)
                vystup.append(temp / (1 + temp))
        return np.array(vystup)

    def deriv(self):
        temp = self.forward(self.vstup)
        return temp * (1 - temp)

    def back(self, chyba, koeficientUceni):
        return np.multiply(self.deriv(), chyba)


class SoftMax:
    def forward(self, vstup):
        temp = np.exp(np.array(vstup) - np.max(vstup))
        self.vystup = np.array(temp / np.sum(temp))
        return self.vystup

    def deriv(self):
        return self.vystup * (np.identity(np.size(self.vystup)) - self.vystup).T

    def back(self, chyba, koeficientUceni):
        return np.dot(self.deriv(), chyba)


class ReLu:
    def forward(self, vstup):
        self.vstup = vstup
        return np.maximum(0, vstup)

    def deriv(self):
        return np.array(self.vstup) > 0

    def back(self, chyba, koeficientUceni):
        return np.multiply(self.deriv(), chyba)


class SS:  # jako Skalarni Soucin

    def __init__(self, pocetVstupu, pocetVystupu):  # pocetVystupu je pocet Neuronu
        self.maticeVah = np.random.rand(pocetVstupu + 1, pocetVystupu) - 0.5  # + 1 je bias

    def forward(self, vstup):
        self.vstup = np.append(vstup, 1)
        return np.dot(self.vstup.T, self.maticeVah)

    def back(self, chyba, koeficientUceni):
        chybaRaketak = np.dot(self.maticeVah[:-1], chyba)
        self.maticeVah += koeficientUceni * np.multiply(np.array([self.vstup]).T,
                                                        chyba)  # 'nejde' transponovat vektor ... proto to musi jit na matici
        return chybaRaketak


class NN:
    def __init__(self, sit, koeficientUceni, funkceChyby=None):
        self.sit = sit
        self.koeficientUceni = koeficientUceni
        self.funkceChyby = funkceChyby

    def forward(self, vstup):
        self.vystup = vstup
        for vrstva in self.sit:
            self.vystup = vrstva.forward(self.vystup)
        return self.vystup

    def back(self, pocatecniChyba):
        self.chyba = pocatecniChyba
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

# TODO otestovat rozdil rychlosti s ne numpy NN na MNISTu
