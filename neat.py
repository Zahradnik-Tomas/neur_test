# >vyfiltrovan lepsimi algoritmy jako ES-HyperNEAT a EANT2 ... https://web.archive.org/web/20240224165757/https://raw.githubusercontent.com/iTODDLERS-BTFO/iToddlers-BTFO/master/satania/smug1.jpg

import random

import numpy as np


class Node:
    def __init__(self, hodnota=0.0, hidden=True, input=False, bias=False):
        self.hodnota = hodnota
        self.SynapsyListOut = []
        self.SynapsyListInp = []
        self.krokID = 0
        self.hidden = hidden
        self.input = input
        self.bias = bias
        self.hledaInput = False
        self.ukoncenoHledani = 0


class Manager:  # TODO druhy, crossover, vymysleni a domysleni toho algoritmu (mohu tomu vubec rikat NEAT?)
    def __init__(self, pocetInputu, pocetOutputu):
        self.pocetInputu = pocetInputu
        self.pocetOutputu = pocetOutputu
        self.dicSynaps = {}
        self.listSiti = []  # TODO listSiti nebo listDruhu nebo oboje?, listDruhu musi byt dikcionar
        self.id = 0
        self.nejvyssi_node = 0
        self.SanceMutaceVahSpecifickeSynapsy = 0.2
        self.SanceMutaceSynaps = 0.2
        self.SanceMutacePridaniNodu = 0.2
        self.SanceMutacePridaniSynapsy = 0.2
        self.SanceVypnutiSynapsy = 0.05

    def PridejEntry(self, input, output):
        self.nejvyssi_node = max(self.nejvyssi_node, input, output)
        if self.KontrolaEntry(input, output):
            self.dicSynaps[f"{input}-{output}"] = Entry(input, output, self.id)
            self.id += 1
            return self.id - 1
        else:
            return self.dicSynaps[f"{input}-{output}"].id

    def KontrolaEntry(self, input, output):
        return f"{input}-{output}" not in self.dicSynaps  # https://wiki.python.org/moin/TimeComplexity#dict

    def ZmutujSit_PridejSynapsu(self, sit):  # Buh roni slzy
        listus = list(sit.listNodu.keys())
        while len(listus) > 0:
            node1 = random.choice(listus)
            listus.remove(node1)
            if not sit.MoznostNoveKonekce(node1):
                continue
            listMoznosti = set(listus)
            if node1 in sit.listKonekci:
                dalsiList = set(sit.listKonekci[node1])
                moznosti = list(listMoznosti.difference(dalsiList))
            else:
                moznosti = listus.copy()
            while len(moznosti) > 0:
                node2 = random.choice(moznosti)
                moznosti.remove(node2)
                idS = self.PridejEntry(node1, node2)
                if sit.PridejSynapsu(Synapsa(node1, node2, random.random() - 0.5, idS)) == -1:
                    continue
                return 0
        return -1

    def ZmutujSit_PridejNode(self, sit):  # TODO jeste odeber node? jak budu zmensovat tu sit?
        if len(sit.listSynaps.values()) == 0:  # Tom tu, nebudu odebirat nody, budu odebirat vypnute synapsy potomku crossoveru
            return -1
        synapsa = random.choice(list(
            sit.listSynaps.values()))
        if synapsa.input == synapsa.output:
            return -2
        nodeId = self.nejvyssi_node + 1
        idS = self.PridejEntry(synapsa.input, nodeId)
        idS2 = self.PridejEntry(nodeId, synapsa.output)
        synaps1 = Synapsa(synapsa.input, nodeId, synapsa.vaha, idS)
        synaps2 = Synapsa(nodeId, synapsa.output, 1.0, idS2)
        if sit.PridejSynapsu(synaps1) != -1 and sit.PridejSynapsu(synaps2) != -1:
            synapsa.povolen = False

    def ZmutujSit_ZmutujVahy(self, sit):
        for synapsa in sit.listSynaps.values():
            if random.random() >= self.SanceMutaceVahSpecifickeSynapsy:
                continue
            synapsa.vaha *= random.uniform(0.98, 1.02)

    def ZmutujSit_VypniSynapsy(self, sit):
        for synapsa in sit.listSynaps.values():
            if random.random() >= self.SanceVypnutiSynapsy:
                synapsa.povolen = False

    def CrossOver(self, dominantni, recesivni):
        pass  # TODO

    def VytvorInicialniPopulaci(self, popNum):  # bude to FS-NEAT, ne klasicky NEAT
        for i in range(popNum):
            self.listSiti.append(Sit(self.pocetInputu, self.pocetOutputu))
            a = random.randint(0, self.pocetInputu)
            b = random.randrange(self.pocetInputu + 1, self.pocetInputu + 1 + self.pocetOutputu)
            idS = self.PridejEntry(a, b)
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
        for node in range(0, self.pocetInputu + 1):
            self.SkutecnyForward(self.listNodu[node], self.krokID)
        output = []
        for i in range(self.pocetInputu + 1, self.pocetOutputu + self.pocetInputu + 1):
            output.append(self.listNodu[i].hodnota)
        self.krokID *= -1
        return output

    def SkutecnyForward(self, node,
                        krokID,
                        otecHledaInput=False):  # je to vice Forward nez Forward, ale chci aby Forward se jmenoval Forward, takze toto bude SkutecnyForward
        if node.krokID == krokID:
            return
        node.krokID = krokID
        if not node.input and node.ukoncenoHledani == 0:
            node.hodnota = 0
        node.hledaInput = True
        for i in range(node.ukoncenoHledani, len(node.SynapsyListInp)):
            Synapsa = node.SynapsyListInp[i]
            if not Synapsa.povolen:
                continue
            if Synapsa.krokID != krokID:
                if self.listNodu[Synapsa.input].hledaInput:
                    node.krokID *= -1
                    node.ukoncenoHledani = i
                    return 2
                tempcislo = self.SkutecnyForward(self.listNodu[Synapsa.input], krokID, True)
                if tempcislo == 2 and otecHledaInput:
                    node.krokID *= -1
                    node.ukoncenoHledani = i
                    return 2
            node.hodnota += Synapsa.hodnota
        node.ukoncenoHledani = 0
        node.hledaInput = False
        if node.hidden:
            node.hodnota = np.tanh(node.hodnota)
        for Synapsa in node.SynapsyListOut:
            Synapsa.krokID = krokID
            if Synapsa.povolen:
                Synapsa.hodnota = Synapsa.vaha * node.hodnota
        for Synapsa in node.SynapsyListOut:  # pokud pujde rovnou do rekurze, tak nastane problem kdy node uz je oznacen jako hotovy, ale jeste nedal hodnotu synapse, kterou pouzivame
            if Synapsa.povolen:
                self.SkutecnyForward(self.listNodu[Synapsa.output], krokID)
        if node.bias:
            node.hodnota = 1.0

    def PridejSynapsu(self, Synapsa):
        if Synapsa.input == Synapsa.output:  # pokud by si treba bias neuron udelal primou konekci sam na sebe, tak by jeho output zacal nekontrolovatelne rust
            return -2
        if Synapsa.id in self.listSynaps:
            return -1
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
        return 0

    def MoznostNoveKonekce(self,
                           input):  # prozatimni vecicka co mi pomuze tvorit nove konekce
        if input not in self.listKonekci and len(self.listNodu) > 0:
            return True
        return len(self.listKonekci[input]) < len(self.listNodu) - 1  # -1 je odebrani moznosti sam na sebe

    def KontrolaNodu(self, nodeID):
        return nodeID in self.listNodu


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
