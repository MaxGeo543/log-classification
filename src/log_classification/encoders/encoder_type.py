from log_classification.encoders.datetime_encoder import DatetimeEncoder
from log_classification.encoders.loglevel_encoder import LogLevelEncoder
from log_classification.encoders.function_encoder import FunctionEncoder
from log_classification.encoders.message_encoder import MessageEncoder
from log_classification.encoders.classes_encoder import ClassesEncoder

class EncoderType:
    """
    Defines the different types of encoders
    """
    
    datetime = "datetime"
    loglevel = "loglevel"
    function = "function"
    message = "message"
    classes = "classes"

    @classmethod
    def types(cls):
        """
        Get a list of all encoder types
        """
        return [
            value for name, value in cls.__dict__.items()
            if not name.startswith('__')
            and not callable(value)
            and not isinstance(value, (classmethod, staticmethod))
        ]

    @classmethod
    def to_str(cls, obj: object):
        """
        Get the string name of the encoder type of an encoder object
        """
        if isinstance(obj, DatetimeEncoder):
            return cls.datetime
        elif isinstance(obj, LogLevelEncoder):
            return cls.loglevel
        elif isinstance(obj, FunctionEncoder):
            return cls.function
        elif isinstance(obj, MessageEncoder):
            return cls.message
        elif isinstance(obj, ClassesEncoder):
            return cls.classes
        else:
            raise ValueError(f"Unknown encoder type: {type(obj)}")

    @classmethod
    def to_type(cls, st: str):
        """
        Get the encoder type using the string name 
        """
        if st == cls.datetime:
            return DatetimeEncoder
        elif st == cls.function:
            return FunctionEncoder
        elif st == cls.loglevel:
            return LogLevelEncoder
        elif st == cls.message:
            return MessageEncoder
        elif st == cls.classes:
            return ClassesEncoder
        else:
            raise ValueError(f"Unknown encoder type: {st}")