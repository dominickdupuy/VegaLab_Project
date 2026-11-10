"""The dashboard server: the static site in ``web/out`` plus a read-mostly JSON API.

Lane: Ilan Danial (machinery).

    python -m optspread.server --port 8600
    python -m optspread.server --runs runs --data "$VEGALAB_DATA" --workers 2

The code now lives in ``optspread.api.server``. This module stays as the entry point: it
re-exports every name it used to define, the gates load the app from it by module path
(``m3gate.SERVER_MODULE``, ``m4gate.SERVER_MODULE``), and it still runs the server when
executed as above.
"""

from optspread.api.server import CACHE_DIR as CACHE_DIR
from optspread.api.server import CACHE_ENV as CACHE_ENV
from optspread.api.server import IMMUTABLE as IMMUTABLE
from optspread.api.server import LOCAL_HOSTS as LOCAL_HOSTS
from optspread.api.server import WEB_DIR as WEB_DIR
from optspread.api.server import Day as Day
from optspread.api.server import OptionalDay as OptionalDay
from optspread.api.server import Settings as Settings
from optspread.api.server import _day as _day
from optspread.api.server import allowed_hosts as allowed_hosts
from optspread.api.server import code_stamp as code_stamp
from optspread.api.server import create_app as create_app
from optspread.api.server import main as main
from optspread.api.server import redactor as redactor

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
