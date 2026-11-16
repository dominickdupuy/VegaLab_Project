"""Plug-in trading agents: your own ``Policy``, benchmarked exactly like PPO, A2C and DQN.

A plug-in is Python on the serving machine, named by an import path, never code sent from
a browser. The path is ``package.module:attr`` (importable from ``sys.path``) or
``path/to/file.py:attr``, and ``attr`` is an ``AlphaPlugin``:

* ``policy``, a zero-argument factory returning a ``Policy`` (``reset(seed)``,
  ``act(obs, info) -> action id``): a rule. It is scored as it stands, like the scripted
  arms: one deterministic run, no training budget, no seeds.
* or ``trainer``, ``Callable[[RunConfig, ChainProvider], Policy]``, the benchmark's own
  ``SynthTrainer`` shape: a learner. It is trained like a learned arm, once per training
  seed on the dataset's train paths and once on the twin's (the noise control), with the
  seed and budget in ``config.train.seed`` and ``config.train.total_timesteps`` and the
  market and episode in ``config.market`` and ``config.env``. ``config.policy`` and
  ``config.train.algorithm`` are placeholders (``ppo``), since ``RunConfig`` names only
  the built-in arms; a plug-in ignores them.

Where plug-ins are listed: ``OPTSPREAD_ALPHA_PLUGINS`` (import paths separated by commas
or whitespace) and the JSON file ``OPTSPREAD_ALPHA_PLUGINS_FILE`` names
(``{"plugins": ["pkg.mod:attr", ...]}``). Both are read by every process that resolves an
arm, so the server's job workers, which inherit its environment, see the same plug-ins.
``register`` adds one in process (the command line's ``--plugin``) and to the environment,
for any process started after it.

A benchmark of a plug-in is keyed by its name and the digest of its module's source file,
so editing the agent's file gives new results rather than serving stale ones. The digest is
kept current: every lookup (``plugins``, ``find``) checks the source file's inode, size and
modification time against the ones it was loaded from, and on a change imports the module
again and digests the new source, in the server and in each long-lived job worker alike.
A digest therefore always names the code that would run now; an edit that no longer
imports is refused with its reason rather than served from the old code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from optspread.config import RunConfig
from optspread.policies import Policy
from optspread.types import ChainProvider

PLUGINS_ENV = "OPTSPREAD_ALPHA_PLUGINS"
PLUGINS_FILE_ENV = "OPTSPREAD_ALPHA_PLUGINS_FILE"
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_NAME = re.compile(NAME_PATTERN)
# The built-in arms, and the scripted policy names a plug-in must not shadow either.
RESERVED = ("ppo", "a2c", "dqn", "delta_seller", "always_on", "flat", "random")

PolicyFactory = Callable[[], Policy]
PluginTrainer = Callable[[RunConfig, ChainProvider], Policy]


class PluginError(ValueError):
    """A plug-in that cannot be loaded, or is not an ``AlphaPlugin``; the message says why."""


@dataclass(frozen=True, slots=True)
class AlphaPlugin:
    """A trading agent for the Synthetic alpha benchmark: a rule (``policy``) or a learner
    (``trainer``), exactly one of them."""

    name: str
    label: str = ""
    policy: PolicyFactory | None = None
    trainer: PluginTrainer | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise PluginError(
                f"plug-in name {self.name!r} must be lower case letters, digits and "
                "underscores, starting with a letter, at most 32 characters"
            )
        if self.name in RESERVED:
            raise PluginError(f"plug-in name {self.name!r} is a built-in arm's")
        if (self.policy is None) == (self.trainer is None):
            raise PluginError(
                f"plug-in {self.name!r} needs exactly one of policy (a rule) and trainer "
                "(a learner)"
            )

    @property
    def trainable(self) -> bool:
        return self.trainer is not None

    @property
    def title(self) -> str:
        return self.label or self.name.replace("_", " ")


Stamp = tuple[int, int, int]  # a file's inode, size and modification time (ns)


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    plugin: AlphaPlugin
    source: str  # the import path it was registered by
    digest: str  # first 8 hex of the sha256 of its module's source file
    file: Path | None = None  # that source file, when it has one
    stamp: Stamp | None = None  # the file's stamp when it was read and imported

    @property
    def name(self) -> str:
        return self.plugin.name

    @property
    def key(self) -> str:
        """What its benchmark directories are named by: the name and the code's digest."""
        return f"{self.plugin.name}.{self.digest}"


def _stamp(file: Path | None) -> Stamp | None:
    if file is None:
        return None
    try:
        st = file.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


# What each plug-in module was last executed from, by module name: its source file's
# stamp and the digest of the very bytes executed. A module whose file has changed since
# is executed again, from its source and never a cached .pyc (which Python trusts on the
# whole second and the size, so a quick edit of the same length would run the old code).
_imported: dict[str, tuple[Stamp | None, str]] = {}


def _source_file(module: ModuleType) -> Path | None:
    try:
        source = inspect.getsourcefile(module)
    except TypeError:
        return None
    return Path(source) if source else None


def _read(file: Path | None) -> tuple[Stamp | None, str]:
    """The file's stamp, then the digest of its bytes (an edit between shows later)."""
    stamp = _stamp(file)
    try:
        data = file.read_bytes() if file is not None else b""
    except OSError:
        data = b""
    return stamp, hashlib.sha256(data).hexdigest()[:8]


def _execute(module: ModuleType, file: Path) -> None:
    """Run ``file``'s source as it is now in ``module``, and record what was run."""
    stamp = _stamp(file)
    data = file.read_bytes()
    exec(compile(data, str(file), "exec"), module.__dict__)
    _imported[module.__name__] = (stamp, hashlib.sha256(data).hexdigest()[:8])


def _module(path: str) -> ModuleType:
    """The module ``path`` names, executed again first if its source changed since."""
    if path.endswith(".py") or "/" in path or os.sep in path:
        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise PluginError(f"plug-in file {path!r} does not exist")
        name = "optspread_alpha_plugin_" + hashlib.sha256(str(file).encode()).hexdigest()[:12]
        seen = _imported.get(name)
        if name in sys.modules and seen is not None and seen[0] == _stamp(file):
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise PluginError(f"plug-in file {path!r} cannot be imported")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            _execute(module, file)
        except Exception as exc:
            del sys.modules[name]  # never the old code under the new file's name
            _imported.pop(name, None)
            raise PluginError(f"plug-in file {path!r} failed to import: {exc!r}") from exc
        return module
    try:
        module = importlib.import_module(path)
    except ImportError as exc:
        raise PluginError(f"plug-in module {path!r} cannot be imported: {exc}") from exc
    except Exception as exc:
        raise PluginError(f"plug-in module {path!r} failed to import: {exc!r}") from exc
    source = _source_file(module)
    if module.__name__ not in _imported:
        # imported before (by its package, or at start-up): taken as its file is now
        _imported[module.__name__] = _read(source)
    elif source is not None and _imported[module.__name__][0] != _stamp(source):
        try:
            _execute(module, source)
        except Exception as exc:
            _imported.pop(module.__name__, None)
            raise PluginError(f"plug-in module {path!r} failed to import: {exc!r}") from exc
    return module


def load_plugin(path: str) -> LoadedPlugin:
    """Import ``package.module:attr`` or ``file.py:attr`` and check it is an ``AlphaPlugin``
    (executing its module again if the source changed since it last was)."""
    mod_path, sep, attr = path.strip().rpartition(":")
    if not sep or not mod_path or not attr:
        raise PluginError(
            f"plug-in {path!r} is not an import path: write package.module:attr or "
            "path/to/file.py:attr"
        )
    module = _module(mod_path)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise PluginError(f"plug-in {path!r}: {mod_path} has no attribute {attr!r}") from exc
    if not isinstance(obj, AlphaPlugin):
        raise PluginError(
            f"plug-in {path!r} is a {type(obj).__name__}, not an AlphaPlugin "
            "(optspread.alpha.plugins.AlphaPlugin)"
        )
    # the source the module was executed from: its stamp and digest
    stamp, digest = _imported.get(module.__name__) or _read(_source_file(module))
    return LoadedPlugin(obj, path.strip(), digest, _source_file(module), stamp)


def configured_paths(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The import paths the environment lists, the variable's first, then the file's."""
    e = os.environ if env is None else env
    paths = [p for p in re.split(r"[,\s]+", e.get(PLUGINS_ENV, "")) if p]
    file = e.get(PLUGINS_FILE_ENV, "")
    if file:
        try:
            listed = json.loads(Path(file).expanduser().read_text())["plugins"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PluginError(
                f"{PLUGINS_FILE_ENV}={file!r} is not a JSON file of the form "
                '{"plugins": ["package.module:attr", ...]}'
            ) from exc
        if not isinstance(listed, list) or not all(isinstance(p, str) for p in listed):
            raise PluginError(f"{PLUGINS_FILE_ENV}={file!r}: plugins must be a list of strings")
        paths += listed
    return tuple(dict.fromkeys(paths))


_cache: dict[tuple[str, ...], dict[str, LoadedPlugin]] = {}
_lock = threading.RLock()  # the server resolves arms from its thread pool


def _changed(loaded: LoadedPlugin) -> bool:
    return loaded.stamp != _stamp(loaded.file)


def plugins() -> dict[str, LoadedPlugin]:
    """Every configured plug-in by name, loaded once per configuration and loaded again,
    with a new digest, whenever its source file has changed since."""
    paths = configured_paths()
    with _lock:
        cached = _cache.get(paths)
        if cached is not None and not any(_changed(p) for p in cached.values()):
            return cached
        out: dict[str, LoadedPlugin] = {}
        for p in paths:
            loaded = load_plugin(p)
            if loaded.name in out and out[loaded.name].source != loaded.source:
                raise PluginError(
                    f"two plug-ins are named {loaded.name!r}: {out[loaded.name].source} and "
                    f"{loaded.source}"
                )
            out[loaded.name] = loaded
        _cache[paths] = out
        return out


def find(name: str) -> LoadedPlugin | None:
    """The plug-in called ``name``, or None for a built-in or unknown arm."""
    return None if name in RESERVED else plugins().get(name)


def register(path: str) -> LoadedPlugin:
    """Load ``path`` and add it to ``OPTSPREAD_ALPHA_PLUGINS`` for this process and its children."""
    loaded = load_plugin(path)
    before = os.environ.get(PLUGINS_ENV)
    paths = [p for p in re.split(r"[,\s]+", before or "") if p]
    if loaded.source not in paths:
        os.environ[PLUGINS_ENV] = ",".join([*paths, loaded.source])
    try:
        plugins()  # a clash with one already listed is refused here, not at first use
    except PluginError:
        if before is None:
            os.environ.pop(PLUGINS_ENV, None)
        else:
            os.environ[PLUGINS_ENV] = before
        raise
    return loaded
