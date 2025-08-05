from datetime import datetime, timezone
from encoders.datetime_features import DatetimeFeatureBase, DatetimeFeature
from typing import Any
from hash_list import hash_list_to_string

class DatetimeEncoder:
    def __init__(self, features: list[type[DatetimeFeatureBase]]):
        self.features = features
        
        # calculate dimension
        dt = datetime.now(timezone.utc)
        dimension = 0
        for f in self.features:
            v = f(dt).value
            if isinstance(v, int) or isinstance(v, float):
                dimension += 1
            else:
                dimension += len(v)
        self.dimension = dimension

    def extract_date_time_features(self, dt: datetime) -> dict[str, Any]:
        if dt.tzinfo is None or dt.utcoffset() is None: 
            dt = dt.replace(tzinfo=timezone.utc)

        return {(instance := f(dt)).key: instance.value for f in self.features}

    def get_dimension(self):
        return self.dimension

    def get_key(self):
        key = hash_list_to_string([
            "DatetimeEncoder",
            *[f.key for f in self.features]
        ], 16)
        return key



if __name__ == "__main__":
    x = extract_date_time_features(datetime.now(), [
        DatetimeFeature.day.normalized, DatetimeFeature.month.normalized, DatetimeFeature.year
    ])

    print(x)