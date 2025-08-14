import pickle
from pathlib import Path
from encoders.datetime_encoder import DatetimeEncoder
from encoders.loglevel_encoder import LogLevelEncoder
from encoders.function_encoder import FunctionEncoder
from encoders.message_encoder import MessageEncoder
from encoders.encoder_type import EncoderType

def save_encoder_if_new(encoder: DatetimeEncoder | LogLevelEncoder | FunctionEncoder | MessageEncoder,
                        base_path: str, 
                        timestamp: str,
                        verbose: bool = False) -> tuple[str, str, str]:
    """
    Save `encoder` under base_path/encoders/encoder_name, 
    but only if no existing .pkl file there already contains its key.
    
    Returns the Path to the existing or newly created file.
    """
    encoder_name = EncoderType.to_str(encoder)

    key = encoder.get_key()
    base = Path(base_path)
    dir_path = base / "encoders" / encoder_name
    dir_path.mkdir(parents=True, exist_ok=True)

    # look for any .pkl in this dir whose filename contains [key]
    pattern = f"*[{key}]*.pkl"
    matches = list(dir_path.glob(pattern))
    if matches:
        # already saved
        if verbose: print(f"✔️  `{encoder_name}` encoder with key={key!r} already exists at {matches[0]}")
        return encoder_name, key, str(matches[0].relative_to(base))
    else:
        # build new filename and dump
        filename = f"[{key}][{timestamp}].pkl"
        path = dir_path / filename
        with open(path, "wb") as f:
            pickle.dump(encoder, f)
        if verbose: print(f"💾 Saved `{encoder_name}` encoder with key={key!r} to {path}")
        return encoder_name, key, str(path.relative_to(base))