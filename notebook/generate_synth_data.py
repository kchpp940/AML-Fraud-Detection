import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

N = 4000
data = {
    "Timestamp": pd.date_range("2022-01-01", periods=N, freq="min").astype(str),
    "From Bank": np.random.randint(10, 50, N),
    "Account": [f"ACC_{i:05d}" for i in np.random.randint(1, 200, N)],
    "To Bank": np.random.randint(10, 50, N),
    "Account.1": [f"ACC2_{i:05d}" for i in np.random.randint(1, 200, N)],
    "Amount Received": np.random.exponential(1000, N).round(2),
    "Receiving Currency": np.random.choice(["US Dollar", "Euro", "Yuan"], N),
    "Amount Paid": np.random.exponential(1000, N).round(2),
    "Payment Currency": np.random.choice(["US Dollar", "Euro", "Yuan"], N),
    "Payment Format": np.random.choice(["Cheque", "Credit Card", "Wire", "ACH", "Reinvestment"], N),
    "Is Laundering": np.random.choice([0, 1], N, p=[0.95, 0.05]),
}
df = pd.DataFrame(data)

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "HI-Small_Trans.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} synthetic rows to {out_path}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))
