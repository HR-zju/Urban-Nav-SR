from importlib import import_module


def main():
    """Main entry point for the package."""
    server = import_module(".server", __name__)
    server.mcp.run()


def __getattr__(name):
    if name == "server":
        return import_module(".server", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "server"]
