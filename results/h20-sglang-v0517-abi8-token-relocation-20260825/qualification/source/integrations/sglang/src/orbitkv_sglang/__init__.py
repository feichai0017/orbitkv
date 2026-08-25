def register() -> None:
    from .plugin import register as plugin_register

    plugin_register()

__all__ = ["register"]
