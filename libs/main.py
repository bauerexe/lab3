import numpy as np
from sklearn.preprocessing import StandardScaler

from libs.chooseLib import choose_model

if __name__ == '__main__':
    sample_raw = np.asarray(
        [[7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 5]]
    )

    model, scaler = choose_model('alcohol',"torch", "SGD")
    sample_scaled = scaler.transform(sample_raw)
    print(model(sample_scaled))

