import os
import sys
import opengm
import numpy as np
import inspect


data_dir = sys.argv[1]
save_dir = sys.argv[2]
files = os.listdir(data_dir)

print('Processing files in directory:', data_dir)

for f in files:
    graph = []
    gm = opengm.loadGm(os.path.join(data_dir, f))
    print(f, gm.numberOfFactors, gm.numberOfVariables)
    num_factors = gm.numberOfFactors
    for fi in range(gm.numberOfFactors):
        if fi % 50 == 0:
            print(num_factors - fi)
        factor = gm[fi]
        idx = np.array(factor.variableIndices)
        assert idx.shape[0] == 2
        factor_matrix = np.array(factor)
        weight = factor_matrix[idx[0], idx[1]]
        # p = np.exp(weight) / (np.exp(weight) + 1)
        # cost = np.log((1 - p) / p)
        # print(np.unique(factor_matrix), 'cost', weight, 'probability', p)
        # print(cost)
        assert weight == factor_matrix[idx[1], idx[0]]
        graph += [[idx[0], idx[1], weight]]
        # inf = opengm.inference.GraphCut(gm)
        # inf = opengm.inference.BeliefPropagation(gm)
        # inf.infer()
        # arg = inf.arg().reshape(img.shape[0:2])
        # print([idx[0], idx[1], weight])
        # if not np.array(factor.variableIndices).shape[0] == 2:
        #     print(np.array(factor.variableIndices).shape[0])
    graph = np.array(graph)
    save_to = os.path.join(save_dir, f[:-2] + 'npy')
    np.save(save_to, graph)
