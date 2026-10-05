# Генерация синтетических данных (для отладки)
python make_synthetic_data.py --n 400 --seed 42

# Полный пайплайн
python run_all.py --config config.yaml --step all

# Или пошагово
python src/s01_prep.py --config config.yaml
python src/s02_features.py --config config.yaml
python src/s03_diagnostics.py --config config.yaml
python src/s04_preprocess.py --config config.yaml
python src/s05_select_k.py --config config.yaml
python src/s06_cluster.py --config config.yaml
python src/s07_profile.py --config config.yaml --labels kmeans
python src/s08_viz_clusters.py --config config.yaml --interactive
python src/s08_viz_profiles.py --config config.yaml
python src/s09_robust.py --config config.yaml