from sklearn.ensemble import RandomForestClassifier

from dataset_gen import make_classification_dataset
from iterative_sampler import IterativeSampler

# Generowanie przykładowych danych
X, y = make_classification_dataset()

# Tworzenie modelu
model = RandomForestClassifier()

# Inicjalizacja IterativeSampler
sampler = IterativeSampler(model, criterion='hard', sample_size=100, step_size=50, max_iter=10)

# Trenowanie modelu z iteracyjnym próbkowaniem
sampler.fit(X, y)

# Pobieranie historii wyników
history = sampler.get_history()
print("Historia wyników:", history)
