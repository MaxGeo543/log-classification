from datetime import datetime, timezone
from encoders.datetime_features import DatetimeFeatureBase, DatetimeFeature, DT_DICT
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

    def __getstate__(self):
        # Copy normal state but drop/replace the unpicklable object
        state = self.__dict__.copy()
        feats = self.__dict__.get('features')
        state['feature_keys'] = tuple(dtf.key for dtf in (feats or ()))
        state['features'] = None
        return state

    def __setstate__(self, state):
        keys = state.pop('feature_keys', ())
        self.__dict__.update(state)
        # Rehydrate the unpicklable object from a registry/factory
        self.features = [DT_DICT[k] for k in keys if k in DT_DICT]
        missing = [k for k in keys if k not in DT_DICT]
        if missing:
            raise Exception(f"Couldn't find all dt feature classes. These could not be found: {missing}")



if __name__ == "__main__":
    pass