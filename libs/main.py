import numpy as np
from sklearn.preprocessing import StandardScaler

from libs.chooseLib import choose_model

if __name__ == '__main__':
    sample_raw = np.asarray(
        [[7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
    )

    model, scaler = choose_model('quality',"torch", "Adam")
    sample = scaler(sample_raw)
    print(model(sample))

