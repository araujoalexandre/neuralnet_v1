
import os, sys
from os.path import join, exists
from collections import defaultdict

def title(eps):
  eps = str(eps)
  eps = eps.replace('.', '_')
  return eps

def main():
  path = sys.argv[1]
  folders = [
    "2019-02-17_20.21.48.5930",
    "2019-02-27_15.24.06.1345",
    "2019-02-27_15.24.08.0467",
    "2019-03-01_02.26.26.8420"
  ]
  results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
  for i, folder in enumerate(folders):
    for attack in ['FastGradientMethod', 'ProjectedGradientDescent', 'Carlini']:
      for norm in [1, 2, 'inf']:
        if attack == "ProjectedGradientDescent" and norm == 1:
          continue
        for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
          if attack == 'Carlini' and eps != 0.05:
            continue
          for sample in [1, 10, 30, 50]:
            if attack == 'FastGradientMethod':
              config = {'attack':attack, 'norm':norm, 'eps':eps, 'sample':sample}
              score_name = 'score_FastGradientMethod_{norm}_{eps}_{sample}.txt'
            elif attack == 'ProjectedGradientDescent':
              config = {'attack':attack, 'norm':norm, 'eps':eps, 'sample':sample}
              score_name = 'score_{attack}_0.3_{eps}_0.02_10_{sample}_{norm}.txt'
            elif attack == 'Carlini':
              config = {'attack':attack, 'sample':sample}
              score_name = 'score_{attack}_40_9_{sample}.txt'
            score_name = score_name.format(**config)
            filename = join(path, "{}_logs".format(folder), score_name)
            if exists(filename):
              # print(filename)
              with open(filename) as f:
                score = float(f.read().strip().split('\t')[1])
            else:
              score = 0.
            if attack == 'FastGradientMethod':
              key = 'fgm{}'.format(norm)
            elif attack == 'ProjectedGradientDescent':
              key = 'pgd{}'.format(norm)
            elif attack == 'Carlini':
              key = 'carlini{}'.format(norm)
            results[eps][sample][key][folder] = score

  for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    for sample in [1, 10, 30, 50]:
      message = []
      for folder in folders:
        for attack in ['fgm', 'pgd', 'carlini']:
          for norm in [1, 2, 'inf']:
            if attack == 'carlini' and norm in [2, 'inf']:
              continue
            if attack == "pgd" and norm == 1:
              continue
            key = '{}{}'.format(attack, norm)
            score = results[eps][sample][key][folder]
            if score == 0: score = -1
            message += ['{:.5f}'.format(score)]
      print(' '.join(message))

if __name__ == '__main__':
  main()


