#!/usr/bin/env python3
"""
Created on Fri Jan 15 20:12:09 2021

@author: Chengxi Luo
"""
#default grid ranking: left-right, top-bottom

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
import json

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

def Euclidean(row, column):
    #Euclidean distance
    #Euclidean(actual_rank[i][0], actual_rank[i][1])
    if row == 0 and column == 0:
        return 0
    return math.sqrt(row**2+column**2)

def Manhattan(row, column):
    #Manhattan distance
    #Manhattan(actual_rank[i][0], actual_rank[i][1])
    return row+column

def top_bot(image_name):
    #left-right, top-bottom
    #top_bot(i)
    return int(image_name.split('_')[1].split('.')[0])

def top_bot_rev(image_name):
    #bottom-top, right-left
    #top_bot_rev(i)
    return (int(image_name.split('_')[1].split('.')[0])-1)*-1

def weighted_default(image_name):
    #weighted default-> weighted left-right, top-bottom
    #weighted_default(i)
    j = top_bot(image_name)+1
    weight = 1/(math.log2(j+1))
    return weight

def middle_position(grid_pos):
    #middle position bias
    #middle_position(actual_rank[i])
    row = grid_pos[0]
    column = grid_pos[1]
    max_col = grid_pos[2]
    mid_dis = abs(int(max_col/2)-column)
    return row+mid_dis

def nearby_principle(grid_pos):
    return min(grid_pos[0], grid_pos[1])

def rank_node(sources, actual_rank, sink = False):
    #Sort sources or vertex by highest actual ranking
    #Sort sink by lowest actual ranking or non-exist
    node_rank = {}
    for i in sources:
        if i not in actual_rank:
            node_rank[i] = float('INF')
        else:
            node_rank[i] = middle_position(actual_rank[i])
    if sink:
        return [k for k, v in node_rank.items() if v == max(node_rank.values())]
    return [k for k, v in node_rank.items() if v == min(node_rank.values())]

def greedy_fas(judgements, actual_rank):
    s1 = []
    s2 = []
    while judgements.nodes != set():
        while judgements.findSinks() != []:
            sinks = judgements.findSinks()
            u = random.choice(rank_node(sinks, actual_rank, sink = True))
            s2.insert(0, u)
            judgements.remove_node(u)

        while judgements.findSources() != []:
            sources = judgements.findSources()
            u = random.choice(rank_node(sources, actual_rank))

            s1.append(u)
            judgements.remove_node(u)

        vertexs = judgements.getMostDeltaDegreeNodes()
        if vertexs != []:
            u = random.choice(rank_node(vertexs, actual_rank))
            s1.append(u)
            judgements.remove_node(u)
    return s1+s2

def grid_to_list(actual_rank, ideal):
    actual_list = []
    node_rank = {}
    for i in actual_rank:
        node_rank[i] = middle_position(actual_rank[i])
    for i in actual_rank:
        min_list = [k for k, v in node_rank.items() if v == min(node_rank.values())]
        m = float('INF')
        node = None
        for j in min_list:
            rank = ideal.index(j)
            if rank < m:
                m = rank
                node = j
        actual_list.append(node)
        del(node_rank[node])
    return actual_list

def open_standard_file(filename):
    standard = []
    with open(filename, encoding='utf-8', mode = 'r') as f:
        for line in f:
            jObj = line.strip().split("\t")
            standard.append(jObj)
    return standard[1:]

def write_csvfile(filename, content):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')

        for i in content:
            writer.writerow(i)

def open_image_pairs(filename):
    jud = {}
    with open(filename, encoding='utf-8', mode = 'r') as f:
        for line in f:
            jObj = line.strip().split("\t")
            pair = jObj[1].split(',')
            currentTopic = jObj[0]
            if currentTopic == 'Query':
                continue
            for i in [jObj[2], jObj[3], jObj[4]]:
                if i in ['-1','-2']:
                    jud[currentTopic] = jud.setdefault(currentTopic, graph()).add_edge(pair[0], pair[1])
                elif i in ['1','2']:
                    jud[currentTopic] = jud.setdefault(currentTopic, graph()).add_edge(pair[1], pair[0])
                #not build edge in the tie situation
    return jud


def open_image_pos(filename):
    pos = {}
    with open(filename) as f:
        jObj = json.load(f)
        for topic, search_engines in jObj.items():
            pos[topic] = {}
            for engine, image_pos in search_engines.items():
                if engine == '0':
                    search_engine_name = 'sogou/'
                elif engine == '1':
                    search_engine_name = 'baidu/'
                for image in image_pos:
                    full_name = search_engine_name + topic + '_' + image + '.jpg'
                    pos[topic][full_name] = image_pos[image]
    return pos

def separate_rank(rank):
    sogou = []
    baidu = []
    for i in rank:
        if i[:5] == 'baidu':
            baidu.append(i)
        else:
            sogou.append(i)
    return sogou, baidu

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='Ranking by greedy feedback arc set')
    parser.add_argument('-p', type=float, default=0.95, help='persistence')
    parser.add_argument('prefs', type=str, help='Preferences judgments')
    parser.add_argument('run', type=str, help='Actual search results')
    args = parser.parse_args()

    hiQ_filename = args.prefs
    actualrank_filename = args.run


    p=0.95
    judgements_graph = open_image_pairs(hiQ_filename)

    currentTopic = ''
    actual_rank = open_image_pos(actualrank_filename)

    s_total = 0.0
    b_total = 0.0
    N = 0
    topic_num = 2
    fas_rank = {}


    winner = []


    #print('Start computing ideal ranking.')
    for topic in judgements_graph:
        if topic in actual_rank:
            rank = greedy_fas(judgements_graph[topic], actual_rank[topic])
            fas_rank[topic] = rank

            actual = grid_to_list(actual_rank[topic],rank)
            s_actual,b_actual = separate_rank(actual)

            s_score = rbo(rank, s_actual, float(p))
            print('sogou', topic_num, s_score, sep=',')
            b_score = rbo(rank, b_actual, float(p))
            print('baidu', topic_num, b_score, sep=',')

            topic_num += 1
        N += 1

    if N > 0:
        print('sogou', 'amean', s_total/N, sep=',')
        print('baidu', 'amean', b_total/N, sep=',')
    else:
        print('sogou', 'amean, 0.0')
        print('baidu', 'amean, 0.0')
