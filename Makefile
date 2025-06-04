.PHONY: notebooks tests

notebooks:
jupyter nbconvert --to notebook --execute notebook/data_wrangling.ipynb
jupyter nbconvert --to notebook --execute notebook/model_training.ipynb

tests:
pytest -q
