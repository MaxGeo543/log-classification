from __future__ import annotations

from datetime import datetime, date, timezone, timedelta
import calendar
import numpy as np
from abc import ABC
import sys

_epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)


class DatetimeFeatureBase(ABC):
    """
    Abstract base class for all datetime features.

    Stores a single feature value derived from a datetime and provides
    an interface for subclasses to define feature-specific behavior.
    """
    key = ""
    
    def __init__(self, value):
        """
        Initialize the datetime feature with a value.

        :params value: The computed feature value.
        """
        self.value = value

class NormalizedDatetimeFeatureBase:
    """
    Provides extensions for generating normalized datetime features.

    Subclasses gain a `.normalized` class that maps the original feature
    into the [0, 1] range using the feature's min and max values.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Automatically adds a `normalized` subclass upon inheritance.
        """
        super().__init_subclass__(**kwargs)
        normalized_cls = cls._make_normalized_class()
        cls.normalized = normalized_cls
        return cls

    @classmethod
    def _make_normalized_class(cls):
        """
        Dynamically creates the `normalized` feature class.

        :returns: Type[DatetimeFeatureBase]: A class that computes the normalized feature.
        """
        class normalized(DatetimeFeatureBase):
            """
            Normalizes the feature into range [0-1]
            """
            key = cls.key + "_normalized"
            
            def __init__(self, dt: datetime):
                """
                Initialize the normalized version of the feature.

                :params dt: The datetime to compute the feature from.

                :raises Exception: If required attributes are not set on the original feature.
                """
                x = cls(dt)

                if not hasattr(x, "value") or not hasattr(x, "min_value") or not hasattr(x, "max_value"):
                    raise Exception("Objects of classes inheriting NormalizedDatetimeFeatures need to have the attributs `value`, `min_value`, `max_value`  after __init__")

                if isinstance(x.value, tuple):
                    value = tuple(self._normalize(v, x.min_value, x.max_value) for v in x.value)
                else: 
                    value = self._normalize(x.value, x.min_value, x.max_value)

                DatetimeFeatureBase.__init__(self, value)
            
            @staticmethod
            def _normalize(value, min_val, max_val):
                """
                Normalize a value to the [0, 1] range.

                :params value:  The value to normalize.
                :params min_val: The minimum possible value.
                :params max_val: The maximum possible value.

                :returns: float: The normalized value.
                """
                return (value - min_val) / (max_val - min_val)

        return normalized

    class normalized(DatetimeFeatureBase):
        """
        Normalizes the feature into range [0-1]
        """
        pass

class CyclicDatetimeFeatureBase: 
    """
    Provides extensions for generating cyclic datetime features.

    Subclasses gain a `.cyclic` class that maps values into sine and cosine
    pairs to preserve periodicity (e.g., hours, days of the week).
    """
    def __init_subclass__(cls, **kwargs):
        """
        Automatically adds a `cyclic` subclass upon inheritance.
        """
        super().__init_subclass__(**kwargs)
        cyclic_cls = cls._make_cyclic_class()
        cls.cyclic = cyclic_cls
        return cls
    
    @classmethod
    def _make_cyclic_class(cls):
        """
        Dynamically creates the `cyclic` feature class.

        :returns: Type[DatetimeFeatureBase]: A class that computes cyclically encoded features.
        """
        class cyclic(DatetimeFeatureBase, NormalizedDatetimeFeatureBase):
            """
            Represents cyclic datetime features by transforming them
            using sin and cos. This way the cyclic nature of the feature is maintained.

            Note that this will make the value of this class a tuple. 
            """
            key = cls.key + "_cyclic"

            min_value = -1
            max_value = 1

            def __init__(self, dt: datetime):
                """
                Initialize the cyclically encoded version of the feature.

                :params dt: The datetime to compute the feature from.

                :raises Exception: If required attributes are not set on the original feature.
                """
                x = cls(dt)

                if not hasattr(x, "value") or not hasattr(x, "min_value") or not hasattr(x, "max_value"):
                    raise Exception("Objects of classes inheriting NormalizedDatetimeFeatures need to have the attributs `value`, `min_value`, `max_value` after __init__")


                value = self._cyclic(x.value, x.min_value, x.max_value)
                DatetimeFeatureBase.__init__(self, value)
            
            @staticmethod
            def _cyclic(value, min_val, max_val):
                """
                Transform a value into a 2D cyclic representation (sin, cos).

                :params value: The value to transform.
                :params min_val: The minimum of the range.
                :params max_val: The maximum of the range.

                :returns: Tuple[float, float]: The (sin, cos) encoded representation.
                """
                period = max_val - min_val + 1
                normalized = (value - min_val) / period
                angle = 2 * np.pi * normalized
                return np.sin(angle), np.cos(angle)
            
        return cyclic
    
    class cyclic(DatetimeFeatureBase, NormalizedDatetimeFeatureBase): 
        """
        Represents cyclic datetime features by transforming them
        using sin and cos. This way the cyclic nature of the feature is maintained.

        Note that this will make the value of this class a tuple. 

        Can be further normalized using `.normalized` to get features in the range [0-1]
        """
        pass

class SinceMidnightDatetimeFeatureBase: 
    """
    Provides an extension to generate features representing time since midnight.

    Subclasses gain a `.since_midnight` class that computes the duration
    since the start of the day (00:00:00).
    """
    def __init_subclass__(cls, **kwargs):
        """
        Automatically adds a `since_midnight` subclass upon inheritance.
        """
        super().__init_subclass__(**kwargs)
        since_midnight_cls = cls._make_since_midnight_class()
        cls.since_midnight = since_midnight_cls
        return cls
    
    @classmethod
    def _make_since_midnight_class(cls):
        """
        Dynamically creates the `since_midnight` feature class.

        :returns: Type[DatetimeFeatureBase]: A class that computes time since midnight.
        """
        class since_midnight(DatetimeFeatureBase):
            """
            Number of units since midnight.
            """
            key = cls.key + "_since_midnight"
            
            def __init__(self, dt: datetime):
                """
                Initialize the feature representing time since midnight.

                :params dt: The datetime to compute from.

                :raises Exception: If `_calculate_since_midnight` method is not defined in subclass.
                """
                if hasattr(cls, "_calculate_since_midnight"):
                    value = cls._calculate_since_midnight(dt)
                else:
                    raise Exception("Invalid class type for 'since_epoch'")
                
                DatetimeFeatureBase.__init__(self, value)
        return since_midnight

    class since_midnight(DatetimeFeatureBase):
        """
        Number of units since midnight.
        """
        pass

class SinceEpochDatetimeFeatureBase: 
    """
    Provides an extension to generate features representing time since the Unix epoch.

    Subclasses gain a `.since_epoch` class that computes the duration since
    January 1, 1970 (Unix epoch).
    """
    def __init_subclass__(cls, **kwargs):
        """
        Automatically adds a `since_epoch` subclass upon inheritance.
        """
        super().__init_subclass__(**kwargs)
        since_epoch_cls = cls.make_since_epoch_class()
        cls.since_epoch = since_epoch_cls
        return cls
    
    @classmethod
    def make_since_epoch_class(cls):
        """
        Dynamically creates the `since_epoch` feature class.

        :returns: Type[DatetimeFeatureBase]: A class that computes time since epoch.
        """
        class since_epoch(DatetimeFeatureBase):
            """
            Number of units since the Unix Epoch January 1, 1970 (Unix epoch).
            """
            key = cls.key + "_since_epoch"
            
            def __init__(self, dt: datetime):
                """
                Initialize the feature representing time since epoch.

                :params dt: The datetime to compute from.

                :raises Exception: If `_calculate_since_epoch` method is not defined in subclass.
                """
                if hasattr(cls, "_calculate_since_epoch"):
                    value = cls._calculate_since_epoch(dt)
                else:
                    raise Exception("Invalid class type for 'since_epoch'")
                
                DatetimeFeatureBase.__init__(self, value)
        return since_epoch
    
    class since_epoch(DatetimeFeatureBase):
        """
        Number of units since the Unix Epoch January 1, 1970 (Unix epoch).
        """
        pass


class DatetimeFeature:
    """
    Provides multiple fature extraction strategies for datetime features. 
    """
    class day(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase):
        """
        Represents the day component of a datetime object as a feature.

        This class extracts the day of the month (0-indexed) and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        """
        key = "day"

        min_value = 0

        def __init__(self, dt: datetime):
            """
            Initialize the day feature from a datetime object.

            :params dt: The datetime object to extract the day from.
            """
            self.max_value = calendar.monthrange(dt.year, dt.month)[1] - 1
            DatetimeFeatureBase.__init__(self, dt.day - 1)

        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the number of days since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of days since epoch.
            """
            return (dt - _epoch).days

    class week(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase):
        """
        Represents the calendar week of a datetime object as a feature.

        This class extracts the calendar week (0-indexed) and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        """
        key = "week"

        min_value = 0

        def __init__(self, dt: datetime):
            """
            Initialize the week feature from a datetime object.

            :params dt: The datetime object to extract the week from.
            """
            
            self.max_value = date(dt.year, 12, 28).isocalendar()[1] - 1
            super().__init__(dt.isocalendar()[1] - 1)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the number of weeks since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of weeks since epoch.
            """
            days_since_epoch = (dt - _epoch).days
            return days_since_epoch / 7

    class month(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase):
        """
        Represents the month component of a datetime object as a feature.

        This class extracts the month (0-indexed) and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        """
        key = "month"

        min_value = 0
        max_value = 11

        def __init__(self, dt: datetime):
            """
            Initialize the month feature from a datetime object.

            :params dt: The datetime object to extract the month from.
            """
            super().__init__(dt.month - 1)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the number of months since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of months since epoch.
            """
            days_since_epoch = (dt - _epoch).days
            years_since_epoch = days_since_epoch / 365.2425
            return years_since_epoch * 12
        
        class day_count(DatetimeFeatureBase, NormalizedDatetimeFeatureBase):
            """
            Represents the number of days in a month as a feature.

            This class extracts the number of days in a month (28, 29, 30 or 31) and supports
            normalization using:
            
            - `normalized` scaling via `NormalizedDatetimeFeatureBase`
            """
            
            key = "days_in_month"

            min_value = 28
            max_value = 31
            
            def __init__(self, dt: datetime):
                """
                Initialize the days in month feature from a datetime object.

                :params dt: The datetime object to extract the days in month from.
                """

                super().__init__(calendar.monthrange(dt.year, dt.month)[1])

    class year(DatetimeFeatureBase, SinceEpochDatetimeFeatureBase):
        """
        Represents the year component of a datetime object as a feature.

        This class extracts the year and supports
        following datetime feature transformations:
        
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        """
        key = "year"

        def __init__(self, dt: datetime):
            """
            Initialize the year feature from a datetime object.

            :params dt: The datetime object to extract the year from.
            """
            super().__init__(dt.year)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the number of years since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of years since epoch.
            """
            days_since_epoch = (dt - _epoch).days
            return days_since_epoch / 365.2425
        
        class weeks_count(DatetimeFeatureBase, NormalizedDatetimeFeatureBase):
            """
            Represents the number of calendar weeks in a year as a feature.

            This class extracts the number of calendar weeks in a year (52 or 53) and supports
            normalization using:
            
            - `normalized` scaling via `NormalizedDatetimeFeatureBase`
            """
            key = DatetimeFeatureBase.key + "weeks_in_year"

            min_value = 52
            max_value = 53

            def __init__(self, dt: datetime):
                """
                Initialize the number of calendar weeks in a year feature from a datetime object.

                :params dt: The datetime object to extract the days in month from.
                """
                
                super().__init__(date(dt.year, 12, 28).isocalendar()[1])
    
    class weekday(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase):
        """
        Represents the weekday component of a datetime object as a feature.

        This class extracts the weekday (0-indexed) and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        """
        key = "weekday"

        min_value = 0
        max_value = 6

        def __init__(self, dt: datetime):
            """
            Initialize the weekday feature from a datetime object.

            :params dt: The datetime object to extract the weekday from.
            """
            super().__init__(dt.weekday())
    
    class hour(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase, SinceMidnightDatetimeFeatureBase):
        """
        Represents the hour component of a datetime object as a feature.

        This class extracts the hour and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        - `since_midnight` conversion via `SinceMidnightDatetimeFeatureBase`
        """
        key = "hour"

        min_value = 0
        max_value = 23

        def __init__(self, dt: datetime):
            """
            Initialize the hour feature from a datetime object.

            :params dt: The datetime object to extract the hour from.
            """
            super().__init__(dt.hour)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the hours since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of hours since epoch.
            """
            return (dt - _epoch).total_seconds() / 3600
        
        @classmethod
        def _calculate_since_midnight(cls, dt: datetime):
            """
            Calculate the number of hours since midnight for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of hours since midnight.
            """
            return dt.hour + dt.minute / 60 + dt.second / 3600
    
    class minute(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase, SinceMidnightDatetimeFeatureBase):
        """
        Represents the minute component of a datetime object as a feature.

        This class extracts the minute and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        - `since_midnight` conversion via `SinceMidnightDatetimeFeatureBase`
        """
        key = "minute"

        min_value = 0
        max_value = 59

        def __init__(self, dt: datetime):
            """
            Initialize the minute feature from a datetime object.

            :params dt: The datetime object to extract the minute from.
            """
            super().__init__(dt.minute)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the minutes since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of minutes since epoch.
            """
            return (dt - _epoch).total_seconds() / 60

        @classmethod
        def _calculate_since_midnight(cls, dt: datetime):
            """
            Calculate the number of minutes since midnight for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of minutes since midnight.
            """
            return dt.hour * 60 + dt.minute + dt.second / 60
    
    class second(DatetimeFeatureBase, CyclicDatetimeFeatureBase, NormalizedDatetimeFeatureBase, SinceEpochDatetimeFeatureBase, SinceMidnightDatetimeFeatureBase):
        """
        Represents the second component of a datetime object as a feature.

        This class extracts the second and supports
        multiple datetime feature transformations including:
        
        - `cyclic` encoding via `CyclicDatetimeFeatureBase`
        - `normalized` scaling via `NormalizedDatetimeFeatureBase`
        - `since_epoch` conversion via `SinceEpochDatetimeFeatureBase`
        - `since_midnight` conversion via `SinceMidnightDatetimeFeatureBase`
        """
        key = "second"

        min_value = 0
        max_value = 59

        def __init__(self, dt: datetime):
            """
            Initialize the second feature from a datetime object.

            :params dt: The datetime object to extract the second from.
            """
            super().__init__(dt.second)
        
        @classmethod
        def _calculate_since_epoch(cls, dt: datetime):
            """
            Calculate the seconds since the Unix epoch for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of seconds since epoch.
            """
            return (dt - _epoch).total_seconds()

        @classmethod
        def _calculate_since_midnight(cls, dt: datetime):
            """
            Calculate the number of seconds since midnight for the given datetime.

            :params dt: The datetime to calculate from.

            :returns: int: Number of seconds since midnight.
            """
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        

    class is_weekend(DatetimeFeatureBase):
        """
        Represents a feature determinig if the day is on a weekend.

        This class extracts whether the day from a datetime object is on a weekend:
        """
        key = "is_weekend"

        def __init__(self, dt: datetime):
            """
            Initialize the is_weekend feature from a datetime object.

            :params dt: The datetime object to extract the is_weekend feature from.
            """
            weekday = dt.weekday()
            super().__init__(int(weekday >= 5))  

    class is_leap_year(DatetimeFeatureBase):
        """
        Represents a feature determinig if the year is a leap year.

        This class extracts whether the year from a datetime object is a leap year:
        """
        key = "is_leap_year"

        def __init__(self, dt: datetime):
            """
            Initialize the is_leap_year feature from a datetime object.

            :params dt: The datetime object to extract the is_leap_year feature from.
            """
            super().__init__(int(calendar.isleap(dt.year)))


def _get_all_dt_classes():
    def nested_classes_immediate(cls):
        base = cls.__qualname__
        base_depth = base.count('.')
        return {
            obj for obj in cls.__dict__.values()
            if isinstance(obj, type)
            and getattr(obj, "__qualname__", "").startswith(base + ".")
            and obj.__qualname__.count('.') == base_depth + 1
        }

    def get_attached_classes(cls):
        att = dict()

        if issubclass(cls, NormalizedDatetimeFeatureBase):
            x = cls.normalized
            att[x.key] = x
        if issubclass(cls, CyclicDatetimeFeatureBase):
            x = cls.cyclic
            att[x.key] = x
            att[x.normalized.key] = x.normalized
        if issubclass(cls, SinceMidnightDatetimeFeatureBase):
            x = cls.since_midnight
            att[x.key] = x
        if issubclass(cls, SinceEpochDatetimeFeatureBase):
            x = cls.since_epoch
            att[x.key] = x
        
        return att

    DT = DatetimeFeature
    dt_classes = dict()

    immediate = {dtf.key: dtf for dtf in nested_classes_immediate(DT)}
    
    for k, dtf in immediate.items():
        dt_classes.update(get_attached_classes(dtf))
    
    dt_classes.update(immediate)

    return dt_classes

    


DT_DICT = _get_all_dt_classes()

if __name__ == "__main__":
    print(DT_DICT["hour_normalized"](datetime.now()).value)
    
    print(len(DT_DICT))
    for k in DT_DICT: print(k)