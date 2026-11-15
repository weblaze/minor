"""Front matter shared by the methods under methods/.

  cli       command-line parser (--config --seed --tag --code --smoke), config setup, run names
  data      MERT layers and splits, audio codes, art datasets, synthetic smoke data, window sampling
  features  DINOv2 builder, features of a dataset
  scores    structure and distribution scores, genre leakage

Method-specific logic (losses, variants, method-specific code transforms) stays in the method folder.
"""
