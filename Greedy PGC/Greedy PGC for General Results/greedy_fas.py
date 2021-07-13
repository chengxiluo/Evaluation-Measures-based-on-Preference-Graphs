#!/usr/bin/env python3
"""
Created on Fri Jan 15 20:12:09 2021

@author: Chengxi Luo
"""

DEPTH = 1000

def rbo(run, ideal, p):
    run_set = set()
    ideal_set = set()

    score = 0.0
    normalizer = 0.0
    weight = 1.0
    for i in range(DEPTH):
        if i < len(run):
            run_set.add(run[i])
        if i < len(ideal):
            ideal_set.add(ideal[i])
        score += weight*len(ideal_set.intersection(run_set))/(i + 1)
        normalizer += weight
        weight *= p
    return score/normalizer

import random
import math
import csv
import argparse
import re

class graph():
    def __init__(self):
        self.indict = {}
        self.outdict = {}
        self.nodes = set()

    def add_edge(self, f, t):
        self.outdict[f] = self.outdict.setdefault(f, nodeInfo()).add(t)
        self.indict[t] = self.indict.setdefault(t, nodeInfo()).add(f)
        self.nodes.add(f)
        self.nodes.add(t)
        return self

    def remove_node(self, node):
        for in_node in self.getInNodes(node):
            self.outdict[in_node].remove(node)
        if node in self.indict:
            self.indict.pop(node)

        for out_node in self.getOutNodes(node):
            self.indict[out_node].remove(node)
        if node in self.outdict:
            self.outdict.pop(node)

        self.nodes.remove(node)
        return self

    def getInDegree(self, node):
        if node not in self.nodes:
            raise ValueError()
        if node not in self.indict:
            return 0
        return self.indict[node].degree

    def getOutDegree(self, node):
        if node not in self.nodes:
            raise ValueError()
        if node not in self.outdict:
            return 0
        return self.outdict[node].degree

    def getInNodes(self, node):
        if node not in self.nodes:
            raise ValueError()
        if node not in self.indict:
            return  []
        return self.indict[node].nodes

    def getOutNodes(self, node):
        if node not in self.nodes:
            raise ValueError()
        if node not in self.outdict:
            return  []
        return self.outdict[node].nodes

    def getMostDeltaDegreeNodes(self):
        max = float('-inf')
        nodes = []
        for n in self.nodes:
            delta = self.getOutDegree(n) - self.getInDegree(n)
            if delta > max:
                nodes = [n]
                max = delta
            elif delta == max:
                nodes.append(n)
        return nodes

    def findSources(self):
        sources = []
        for n in self.nodes:
            if self.getInDegree(n) == 0:
                sources.append(n)
        return sources

    def findSinks(self):
        sinks = []
        for n in self.nodes:
            if self.getOutDegree(n) == 0:
                sinks.append(n)
        return sinks

    def readFromCSV(self, path, from_col = 0, to_col = 1, sep = ','):
        with open(path) as f:
            for line in f:
                l = line.strip().split(sep)
                self.add_edge(l[from_col], l[to_col])
        return self

class nodeInfo():
    def __init__(self):
        self.degree = 0
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)
        self.degree += 1
        return self

    def remove(self, node):
        self.nodes.remove(node)
        self.degree -= 1
        return self

    def remove_all(self, node):
        while node in self.nodes:
            self.nodes.remove(node)
            self.degree -= 1
        return self

    def degree(self):
        return self.degree

def rank_node(sources, actual_rank, sink = False):
    #Sort sources or vertex by highest actual ranking
    #Sort sink by lowest actual ranking or non-exist
    node_rank = {}
    for i in sources:
        if i not in actual_rank:
            node_rank[i] = float('INF')
        else:
            node_rank[i] = actual_rank[i]
    return sorted(node_rank.items(), key = lambda item:item[1], reverse = sink)[0][0]

def greedy_fas(judgements, actual_rank):
    s1 = []
    s2 = []
    while judgements.nodes != set():
        while judgements.findSinks() != []:
            sinks = judgements.findSinks()
            u = rank_node(sinks, actual_rank, sink = True)
            s2.insert(0, u)
            judgements.remove_node(u)

        while judgements.findSources() != []:
            sources = judgements.findSources()
            u = rank_node(sources, actual_rank)
            s1.append(u)
            judgements.remove_node(u)

        vertexs = judgements.getMostDeltaDegreeNodes()
        if vertexs != []:
            u = rank_node(vertexs, actual_rank)
            s1.append(u)
            judgements.remove_node(u)
    return s1+s2

def open_actual_rank(filename):
    actual_rank = {}
    currentTopic = ''
    with open(filename,mode = 'r') as f:
        for line in f:
            jObj = re.split("\s|(?<!\d)[,.](?!\d)", line.strip())
            currentTopic = jObj[0]
            if currentTopic in actual_rank:
                actual_rank[currentTopic][jObj[2]]=int(jObj[3])
            else:
                actual_rank[currentTopic] = {}
                actual_rank[currentTopic][jObj[2]]=int(jObj[3])
    return actual_rank

def write_csvfile(filename, content_dict):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')

        for topic in content_dict:
            num = 0
            for doc in content_dict[topic]:
                num+=1
                writer.writerow([topic, 0, doc, num])

def readQPrefs(file_name):
    jud = {}
    with open(file_name) as f:
        for line in f:
            jObj = re.split("\s|(?<!\d)[,.](?!\d)", line.strip())
            jud[jObj[0]] = jud.setdefault(jObj[0], graph()).add_edge(jObj[1], jObj[2])
    return jud


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser(
    description='Ranking by greedy feedback arc set')
    parser.add_argument('prefs', type=str, help='Preferences judgments')
    parser.add_argument('run', type=str, help='Actual search results')
    args = parser.parse_args()

    hiQ_filename = args.prefs
    actualrank_filename = args.run'''

    hiQ_filename = 'waterloo.pref'
    actualrank_filename = 'input.clacBase'
    
    #print('Start reading hiQ file:', hiQ_filename)
    judgements_graph = readQPrefs(hiQ_filename)

    currentTopic = ''
    #print('Start reading actual ranking file:', actualrank_filename)
    actual_rank = open_actual_rank(actualrank_filename)


    print('runid,topic,compatibility')
    total = 0.0
    N = 0
    fas_rank = {}
    #print('Start computing ideal ranking.')
    for topic in judgements_graph:
        rank = greedy_fas(judgements_graph[topic], actual_rank[topic])
        fas_rank[topic] = rank
        actual = list(actual_rank[topic].keys())
        actual.sort(key=lambda docno: actual_rank[topic][docno])
        score = rbo(rank, actual, 0.80)
        print(actualrank_filename, topic, score, sep=',')
        total += score
        N += 1
    if N > 0:
        print(actualrank_filename, 'amean', total/N, sep=',')
    else:
        print(actualrank_filename, 'amean, 0.0')

    #Write ideal ranking file
    #write_csvfile(actualrank_filename+'_idealrank', fas_rank)

    #print('Finish writing ideal ranking file:', actualrank_filename+'_idealrank')
