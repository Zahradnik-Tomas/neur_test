# >vyfiltrovan lepsimi algoritmy jako ES-HyperNEAT a EANT2 ... https://web.archive.org/web/20240224165757/https://raw.githubusercontent.com/iTODDLERS-BTFO/iToddlers-BTFO/master/satania/smug1.jpg

import random
import sys

import numpy as np

sys.setrecursionlimit(
    9999999)  # budeme realisti, defaultni limit 1000 by stacil a drive by asi shorelo Slunce (ignorujic nedostatek pameti), nez by ho bylo dosazeno, ale jenom pro jistotu to tu necham


class Node:
    def __init__(self, hodnota=0.0, hidden=True, input=False, bias=False):
        self.hodnota = hodnota
        self.SynapsyListOut = []
        self.SynapsyListInp = []
        self.krokID = 0
        self.hidden = hidden
        self.input = input
        self.bias = bias


class Manager:  # TODO druhy, mutace, crossover, celkove lepsi implementace, nez pokus o ni po precteni papiru bez mysleni o tom
    def __init__(self, pocetInputu, pocetOutputu):  # TODO vymysleni algoritmu novych konekci
        self.pocetInputu = pocetInputu
        self.pocetOutputu = pocetOutputu
        self.dicSynaps = {}
        self.listSiti = []  # TODO listSiti nebo listDruhu nebo oboje?, listDruhu musi byt dikcionar
        self.id = 0

    def PridejEntry(self, input, output):
        if self.KontrolaEntry(input, output):
            self.dicSynaps[f"{input}-{output}"] = Entry(input, output, self.id)
            self.id += 1
            return self.id - 1
        else:
            return self.dicSynaps[f"{input}-{output}"].id

    def KontrolaEntry(self, input, output):
        return f"{input}-{output}" not in self.dicSynaps  # https://wiki.python.org/moin/TimeComplexity#dict

    def ZmutujSit_PridejSynapsu(self, sit):
        pass  # TODO

    def ZmutujSit_PridejNode(self, sit):  # TODO jeste odeber node? jak budu zmensovat tu sit?
        pass  # TODO

    def ZmutujSit_ZmutujVahy(self, sit):
        pass  # TODO

    def CrossOver(self, dominantni, recesivni):
        pass  # TODO

    def VytvorInicialniPopulaci(self, popNum):  # bude to FS-NEAT, ne klasicky NEAT
        for i in range(popNum):
            self.listSiti.append(Sit(self.pocetInputu, self.pocetOutputu))
            a = random.randint(0, self.pocetInputu)
            b = random.randrange(self.pocetInputu + 1, self.pocetInputu + 1 + self.pocetOutputu)
            idS = self.PridejEntry(0, 1)
            self.listSiti[-1].PridejSynapsu(Synapsa(a, b, random.random() - 0.5, idS))


class Sit:
    def __init__(self, pocetInputu, pocetOutputu, napoveda_druhu=-1):
        self.listSynaps = {}  # urceno pro crossover, ve forwardu ironicky pouzivam referenci z nodu
        self.listNodu = {}
        self.listNodu[0] = Node(1.0, False, input=True, bias=True)
        self.pocetInputu = pocetInputu
        self.pocetOutputu = pocetOutputu
        self.krokID = 1
        self.listKonekci = {}
        self.napoveda_druhu = napoveda_druhu

        for i in range(1, pocetInputu + pocetOutputu + 1):
            self.listNodu[i] = Node(hidden=False, input=i <= pocetInputu)

    def Forward(self, vstup):
        for node in range(1, self.pocetInputu + 1):
            self.listNodu[node].hodnota = vstup[node - 1]
        for node in range(0,
                          self.pocetInputu + 1):  # stejny problem, rekurze ale bude chtit hodnotu, kterou jsme jeste nenastavily
            self.ZhotovSynapsyVeForwardu(self.listNodu[node], self.krokID)
        output = []
        for i in range(self.pocetInputu + 1, self.pocetOutputu + self.pocetInputu + 1):
            output.append(self.listNodu[i].hodnota)
        self.krokID *= -1
        return output

    def PridejSynapsu(self, Synapsa):
        if Synapsa.input == Synapsa.output:  # pokud by si treba bias neuron udelal primou konekci sam na sebe, tak by jeho output zacal nekontrolovatelne rust
            return
        self.listSynaps[Synapsa.id] = Synapsa
        if not self.KontrolaNodu(Synapsa.input):
            self.listNodu[Synapsa.input] = Node()
        if not self.KontrolaNodu(Synapsa.output):
            self.listNodu[Synapsa.output] = Node()
        self.listNodu[Synapsa.input].SynapsyListOut.append(Synapsa)
        self.listNodu[Synapsa.output].SynapsyListInp.append(Synapsa)
        if not Synapsa.input in self.listKonekci:
            self.listKonekci[Synapsa.input] = []
        self.listKonekci[Synapsa.input].append(Synapsa.output)

    def MoznostNoveKonekce(self,
                           input):  # prozatimni vecicka co mi pomuze tvorit nove konekce
        if input not in self.listKonekci and len(self.listNodu) > 0:
            return True
        return len(self.listKonekci[input]) < len(self.listNodu) - 1  # -1 je odebrani moznosti sam na sebe

    def KontrolaNodu(self, nodeID):
        return nodeID in self.listNodu

    def ZhotovSynapsyVeForwardu(self, node,
                                krokID):  # je to vice Forward nez Forward, ale chci aby Forward se jmenoval Forward, takze toto bude Zhotoveni SynapsyVeForwardu
        if node.krokID == krokID:
            return
        node.krokID = krokID
        if not node.input:
            node.hodnota = 0
        for Synapsa in node.SynapsyListInp:
            if Synapsa.krokID != krokID:
                self.ZhotovSynapsyVeForwardu(self.listNodu[Synapsa.input], krokID)
            node.hodnota += Synapsa.hodnota
        if node.hidden:
            node.hodnota = np.tanh(node.hodnota)
        for Synapsa in node.SynapsyListOut:
            Synapsa.krokID = krokID
            Synapsa.hodnota = Synapsa.povolen * Synapsa.vaha * node.hodnota
        for Synapsa in node.SynapsyListOut:  # pokud pujde rovnou do rekurze, tak nastane problem kdy node uz je oznacen jako hotovy, ale jeste nedal hodnotu synapse, kterou pouzivame
            self.ZhotovSynapsyVeForwardu(self.listNodu[Synapsa.output], krokID)
        if node.bias:
            node.hodnota = 1.0


class Entry:
    def __init__(self, input, output, id):
        self.input = input
        self.output = output
        self.id = id


class Synapsa:
    def __init__(self, input, output, vaha, id, povolen=True):
        self.input = input
        self.output = output
        self.vaha = vaha
        self.hodnota = 0.0
        self.povolen = povolen
        self.krokID = 0
        self.id = id