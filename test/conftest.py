import os
import tempfile

import pytest


@pytest.fixture
def tmp_fname():
    tfile = tempfile.NamedTemporaryFile(delete=False)
    yield tfile.name
    try:
        os.remove(tfile.name)
    except:
        pass
