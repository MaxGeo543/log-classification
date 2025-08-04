from datetime import datetime, timezone
from datetime_features import DatetimeFeatureBase, DatetimeFeature
from typing import Any

class DatetimeEncoder:
    def __init__(self, features: list[type[DatetimeFeatureBase]]):
        self.features = features
        self.dimension = self.get_dimension()

    def extract_date_time_features(self, dt: datetime) -> dict[str, Any]:
        if dt.tzinfo is None or dt.utcoffset() is None: 
            dt = dt.replace(tzinfo=timezone.utc)

        return {(instance := f(dt)).get_key(): instance.value for f in self.features}

    def get_dimension(self):
        dt = datetime.now(timezone.utc)
        result = 0
        for f in self.features:
            v = f(dt).value
            if isinstance(v, int) or isinstance(v, float):
                result += 1
            else:
                result += len(v)
        
        return v



if __name__ == "__main__":
    x = extract_date_time_features(datetime.now(), [
        DatetimeFeature.day.normalized, DatetimeFeature.month.normalized, DatetimeFeature.year
    ])

    print(x)