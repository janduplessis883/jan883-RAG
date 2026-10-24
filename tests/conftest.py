"""Keep automated tests independent of optional external observability services."""

import os


def pytest_configure(config):
    # The developer environment may export Langfuse credentials for the app.
    # Tests should not create a client or start its background HTTP exporter.
    os.environ["LANGFUSE_DISABLED"] = "1"
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        os.environ.pop(name, None)
