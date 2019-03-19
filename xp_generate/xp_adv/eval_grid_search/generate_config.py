
def title(eps):
  eps = str(eps)
  eps = eps.replace('.', '_')
  return eps

def main():
  config = open('attacks_config.yaml', 'w')
  # carlini config
  with open('carlini_config.txt') as f:
    config.write(f.read())
    config.write('\n\n')
  # pgd_fgsm_config
  with open('pgd_fgsm_config.txt') as f:
    pgd_fgsm_config = f.read()
  # expected_value config
  with open('expected_value_config_carlini.txt') as f:
    expected_value_config = f.read()

  # for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
  #   for norm in ['1', '2', 'INF']:
  #     data = {'eps': eps,
  #             'eps_title': title(eps),
  #             'norm': norm.lower(),
  #             'norm_maj': norm}
  #     config.write(pgd_fgsm_config.format(**data))
  #     config.write('\n')

  # for eps in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
  #   for sample in [1, 10, 30, 50]:
  #     for norm in ['1', '2', 'INF']:
  #       data = {'eps': eps,
  #               'eps_title': title(eps),
  #               'sample': sample,
  #               'norm': norm.lower(),
  #               'norm_maj': norm}
  #       batch_size = 50 // sample
  #       data['batch_size'] = batch_size
  #       config.write(expected_value_config.format(**data))
  #       config.write('\n')

  for sample in [1, 10, 30, 50]:
    batch_size = 50 // sample
    config.write(expected_value_config.format(
      sample=sample, batch_size=batch_size))
    config.write('\n')

  config.close()

if __name__ == '__main__':
  main()
