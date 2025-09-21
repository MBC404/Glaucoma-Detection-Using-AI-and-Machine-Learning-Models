import pandas as pd
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
VF_POINT_DROP_DB = 3        # threshold for per-point worsening
VF_POINT_MIN_COUNT = 3      # minimum number of points to call progression
KEY_SUBJECT = "SUBJECT NUMBER"
KEY_LATERAL = "Laterality"
time_col = "Visit Number"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("merged_glaucoma.csv")

# Identify VF columns
vf_cols = [c for c in df.columns if c.startswith("VF")]
df = df.sort_values([KEY_SUBJECT, KEY_LATERAL, time_col]).reset_index(drop=True)
df[vf_cols] = df[vf_cols].astype(float)

# -----------------------------
# Baseline values
# -----------------------------
baseline_vf = df.groupby([KEY_SUBJECT, KEY_LATERAL])[vf_cols].first()
df = df.merge(
    baseline_vf.add_prefix("baseline_"),
    on=[KEY_SUBJECT, KEY_LATERAL],
    how="left"
)

# Previous visit values
for col in vf_cols:
    df[f"prev_{col}"] = df.groupby([KEY_SUBJECT, KEY_LATERAL])[col].shift(1)

# -----------------------------
# Baseline comparison
# -----------------------------
baseline_worsened = []
for col in vf_cols:
    worsened = (df[col] - df[f"baseline_{col}"] <= -VF_POINT_DROP_DB).astype(int)
    baseline_worsened.append(worsened)
df["event_baseline"] = (np.vstack(baseline_worsened).T.sum(axis=1) >= VF_POINT_MIN_COUNT).astype(int)

# -----------------------------
# Previous visit comparison
# -----------------------------
prev_worsened = []
for col in vf_cols:
    worsened = (df[col] - df[f"prev_{col}"] <= -VF_POINT_DROP_DB).astype(int)
    prev_worsened.append(worsened)
df["event_prev"] = (np.vstack(prev_worsened).T.sum(axis=1) >= VF_POINT_MIN_COUNT).astype(int)

# -----------------------------
# Confirm progression
# -----------------------------
def confirm_progression(group, colname):
    group = group.sort_values(time_col)
    confirmed = (group[colname] & group[colname].shift(1).fillna(0).astype(int))
    return confirmed

df["progression_baseline"] = df.groupby([KEY_SUBJECT, KEY_LATERAL], group_keys=False)\
    .apply(confirm_progression, colname="event_baseline")

df["progression_hybrid"] = (
    df.groupby([KEY_SUBJECT, KEY_LATERAL], group_keys=False)\
      .apply(confirm_progression, colname="event_baseline") |
    df.groupby([KEY_SUBJECT, KEY_LATERAL], group_keys=False)\
      .apply(confirm_progression, colname="event_prev")
).astype(int)

# -----------------------------
# Final carry-forward labels
# -----------------------------
df["Clinical_Status_Baseline"] = df.groupby([KEY_SUBJECT, KEY_LATERAL])["progression_baseline"].cummax()
df["Clinical_Status_Hybrid"] = df.groupby([KEY_SUBJECT, KEY_LATERAL])["progression_hybrid"].cummax()

# -----------------------------
# Keep only original columns + 2 new labels
# -----------------------------
final_cols = [c for c in df.columns if not (
    c.startswith("baseline_") or c.startswith("prev_") or 
    c.startswith("event_") or c.startswith("progression_")
)]
df_final = df[final_cols]

# -----------------------------
# Save results
# -----------------------------
df_final.to_csv("visit_wise_progression_dual_binary.csv", index=False)
print("Saved: visit_wise_progression_dual_binary.csv")
print(df_final[["Clinical_Status_Baseline","Clinical_Status_Hybrid"]].value_counts())