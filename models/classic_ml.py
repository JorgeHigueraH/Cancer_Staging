from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from xgboost import XGBClassifier

class TumorRowlandClassicModels:
    def __init__(self):
        self.log_reg = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        self.rf = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        self.nb = ComplementNB(
            alpha=0.1,
            norm=True
        )
        self.xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )

    def train_all(self, X_train, y_train):
        self.log_reg.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.nb.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

    def predict_all(self, X_test):
        return {
            "logistic": self.log_reg.predict(X_test),
            "random_forest": self.rf.predict(X_test),
            "naive_bayes": self.nb.predict(X_test),
            "xgboost": self.xgb.predict(X_test)
        }

    def predict_proba_all(self, X_test):
        return {
            "logistic": self.log_reg.predict_proba(X_test),
            "random_forest": self.rf.predict_proba(X_test),
            "naive_bayes": self.nb.predict_proba(X_test),
            "xgboost": self.xgb.predict_proba(X_test)
        }