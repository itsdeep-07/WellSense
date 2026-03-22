import joblib
from pathlib import Path
BASE_DIR = Path('.')
try:
    model    = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
    scaler   = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    encoders = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
    features = joblib.load(BASE_DIR / 'models' / 'features.pkl')
    results  = joblib.load(BASE_DIR / 'models' / 'results_df.pkl')
    print("ALL LOADED OK")
except Exception as e:
    import traceback
    traceback.print_exc()
