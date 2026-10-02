import argparse
from dataclasses import fields, MISSING
from typing import get_type_hints, get_origin


def _build_arg_parser(config_class) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    hints = get_type_hints(config_class)
    for f in fields(config_class):
        field_type = hints[f.name]
        is_list = get_origin(field_type) is list or field_type is list
        if f.default_factory is not MISSING:
            default_val = f.default_factory()
        else:
            default_val = f.default
        if default_val is MISSING:
            if is_list:
                parser.add_argument(f"--{f.name}", type=str, nargs="+", required=True)
            else:
                parser.add_argument(f"--{f.name}", type=field_type, required=True)
        elif isinstance(default_val, bool):
            parser.add_argument(
                f"--{f.name}",
                type=lambda v: str(v).lower() not in ("false", "0", "no"),
                default=default_val,
            )
        elif is_list:
            parser.add_argument(f"--{f.name}", type=str, nargs="+", default=default_val)
        else:
            parser.add_argument(f"--{f.name}", type=field_type, default=default_val)
    return parser
