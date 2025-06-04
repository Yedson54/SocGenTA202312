"""Command line interface for energy_predictor."""

import typer
import pandas as pd
from pathlib import Path
from .model import EnergyPredictor

app = typer.Typer(help="Energy consumption reduction predictor")

@app.command()
def predict(data_csv: Path, model_path: Path = Path('model/rf_clf.onnx')) -> None:
    """Run predictions for the given CSV file."""
    df = pd.read_csv(data_csv)
    predictor = EnergyPredictor(model_path)
    predictor.load()
    preds = predictor.predict(df.values)
    for p in preds:
        typer.echo(p)

if __name__ == "__main__":
    app()
