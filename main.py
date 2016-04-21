from perceptron import *
from genetic import *
import numpy as np
import json

if __name__ == "__main__":
    """
    example: Mapped guess prepared Text
    """

    # a = GeneticAlgorithm(GuessText([ 0.88805444,  1.85983713,  1.67446469, 0.50208407,  1.83757417,  1.692993 , 0.82920315, -0.84175092, -2.38020839, -1.0591257 ,  3.40194451, -3.11032175])).run()
    a = GeneticAlgorithm(GuessText([0, 1, 1, 0])).run()


    # b = GuessText([0, 1, 1, 0])
    # # w = b.chromo_to_neural(a[0])
    #
    # # w = b.chromo_to_neural([41.89652763863931, 21.524987862151114, 17.66698405357664, 28.414349238544233, 0.8638186012278546, 0.15101448925410377, 0.05814052856063956, 0.058289509117602334, 0.1144010919874503, 0.9342843009188686, 0.4806748374388562, 0.016639956626580688])
    # nn = NeuralNetwork([2,2,1], 'tanh', w)
    # for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    #     print(i,nn.predict(i))

    # [array([[ 0.88805444,  1.85983713,  1.67446469],
    #    [ 0.50208407,  1.83757417,  1.692993  ],
    #    [ 0.82920315, -0.84175092, -2.38020839]]), array([[-1.0591257 ],
    #    [ 3.40194451],
    #    [-3.11032175]])]
    with open('data.json', 'w') as outfile:
        json.dump(a, outfile)

    pass
