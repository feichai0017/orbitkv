def register() -> None:
    from .hooks import register as hook_register

    hook_register()

__all__ = ("register",)
