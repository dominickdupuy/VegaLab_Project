"""The dashboard's answer cache: a rendered JSON body kept while the files it read stand.

Lane: Ilan Danial (machinery).

The code now lives in ``optspread.api.cache``. This module re-exports every name it used
to define, so existing imports keep working.
"""

from optspread.api.cache import Answers as Answers
from optspread.api.cache import Stamp as Stamp
from optspread.api.cache import answer_key as answer_key
from optspread.api.cache import file_stamp as file_stamp
from optspread.api.cache import tree_stamp as tree_stamp
