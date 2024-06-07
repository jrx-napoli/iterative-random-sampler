# Iterative Random Sampling

#### Classification using random forest
```python main.py --experiment classification --model random_forest_classifier --strategy hard --init_sample_pool_size 0.01 --step_size 100 --max_iter 10 --n_samples 10000 --n_classes 10 --n_features 20 --n_informative 5 --seed 42```

#### Regression using random forest
```python main.py --experiment regression --model random_forest_regressor --strategy hard --init_sample_pool_size 0.01 --step_size 100 --max_iter 10 --n_samples 10000 --n_features 100 --n_informative 10 --seed 42```
