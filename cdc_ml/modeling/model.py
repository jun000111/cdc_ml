from cdc_ml.modeling.calibrate import calibrate_proba


class BookingModel:
    def __init__(self, booster, calibrator, feature_names):
        self.booster = booster  # fitted XGBoost
        self.calibrator = calibrator  # fitted Platt LogisticRegression
        self.feature_names = feature_names

    def predict_proba(self, raw_df):
        X = raw_df[self.feature_names]
        scores = self.booster.predict_proba(X)[:, 1]
        return calibrate_proba(scores, self.calibrator)
