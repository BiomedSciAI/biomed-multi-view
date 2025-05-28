import pandas as pd
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

df_pred = pd.read_csv('outputs/MULTIVIEW_MODEL/pred_MULTIVIEW_MODEL_ft_dreamtarget2035_random_s-101_f-1.0/test_predictions_version_1.csv')

pred_logits = [float(x.strip('[]')) for x in df_pred['prediction'].tolist()]

# sigmoid transform
pred = np.array([sigmoid(x) for x in pred_logits])
# further scale predictions (optional)
pred = (pred - min(pred)) / (max(pred) - min(pred))

# rank predictions
index = np.argsort(pred)[::-1]

# save predictions
# Note that compound IDs are ordered as ID_0, ID_1, ID_2, ... so we can add ID like this
df_top50 = pd.DataFrame({
    'RandomID': ['ID_' + str(x) for x in index[:50].tolist()],
    'Sel_50': 1,
    'Score': pred[index[:50]],
})
df_top50.to_csv('Step2-submission-file.csv', index=False, float_format='%.4f')






