class Accuracy:
    def __init__(self, user_skills, jd_skills):
        self.user_skills = set([s.lower().strip() for s in user_skills])
        self.jd_skills = set([s.lower().strip() for s in jd_skills])

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self._calculate()

    def _calculate(self):
        self.tp = len(self.user_skills & self.jd_skills)
        self.fp = len(self.jd_skills - self.user_skills)
        self.fn = len(self.user_skills - self.jd_skills)
        self.tn = 0

    def precision(self):
        return round(self.tp / (self.tp + self.fp), 2) if (self.tp + self.fp) > 0 else 0.0

    def recall(self):
        return round(self.tp / (self.tp + self.fn), 2) if (self.tp + self.fn) > 0 else 0.0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return round(2 * (p * r) / (p + r), 2) if (p + r) > 0 else 0.0

    def accuracy(self):
        return round(self.tp / (self.tp + self.fp + self.fn), 2) if (self.tp + self.fp + self.fn) > 0 else 0.0

    def report(self):
        return {
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "accuracy": self.accuracy()
        }
