import os
import re

# Regex to match ${VAR_NAME:-DEFAULT_VALUE} or ${VAR_NAME}
PLACEHOLDER_REGEX = re.compile(r"\$\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::-([^}]*))?\s*\}")

def resolve_config_value(value_from_config, default_if_placeholder_not_set=None, target_type=str):
    """
    Resolves a configuration value that might be a placeholder.
    Placeholders can be ${VAR_NAME} or ${VAR_NAME:-Default Value}.
    """
    # Diagnostic print
    if isinstance(value_from_config, str) and "LLAMA_MODEL_TEST" in value_from_config:
        print(f"DEBUG: resolve_config_value called for LLAMA_MODEL_TEST related placeholder: '{value_from_config}'")
        print(f"DEBUG: os.getenv('LLAMA_MODEL_TEST') current value: '{os.getenv('LLAMA_MODEL_TEST')}'")


    if isinstance(value_from_config, str):
        match = PLACEHOLDER_REGEX.fullmatch(value_from_config)
        if match:
            var_name = match.group(1)
            placeholder_default = match.group(2)  # This is None if :-pattern is not used

            env_value = os.getenv(var_name)

            if var_name == "LLAMA_MODEL_TEST": # More specific debug
                print(f"DEBUG: LLAMA_MODEL_TEST: env_value='{env_value}', placeholder_default='{placeholder_default}', function_default='{default_if_placeholder_not_set}'")

            if env_value is not None:
                resolved_value = env_value
            elif placeholder_default is not None:
                resolved_value = placeholder_default
            else:
                resolved_value = default_if_placeholder_not_set
        else:  # Not a placeholder string
            resolved_value = value_from_config
    else:  # Not a string, so not a placeholder
        resolved_value = value_from_config

    if isinstance(value_from_config, str) and "LLAMA_MODEL_TEST" in value_from_config:
         print(f"DEBUG: LLAMA_MODEL_TEST resolved to: '{resolved_value}' before type casting.")


    # Type casting
    if resolved_value is None:
        if target_type is bool:
             return False
        return None

    try:
        if target_type == int:
            return int(resolved_value)
        elif target_type == bool:
            if isinstance(resolved_value, str):
                return resolved_value.lower() in ('true', 'yes', '1', 'on')
            return bool(resolved_value)
        return target_type(resolved_value)
    except ValueError as e:
        print(f"Warning: Could not cast resolved value '{resolved_value}' of type {type(resolved_value)} to {target_type}. Error: {e}.")
        return str(resolved_value) if resolved_value is not None else None
