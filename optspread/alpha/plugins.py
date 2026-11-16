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
so editing the agent's file gives new results rather than serving stale ones.
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


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    plugin: AlphaPlugin
    source: str  # the import path it was registered by
    digest: str  # first 8 hex of the sha256 of its module's source file

    @property
    def name(self) -> str:
        return self.plugin.name

    @property
    def key(self) -> str:
        """What its benchmark directories are named by: the name and the code's digest."""
        return f"{self.plugin.name}.{self.digest}"


def _module(path: str) -> ModuleType:
    if path.endswith(".py") or "/" in path or os.sep in path:
        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise PluginError(f"plug-in file {path!r} does not exist")
        name = "optspread_alpha_plugin_" + hashlib.sha256(str(file).encode()).hexdigest()[:12]
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise PluginError(f"plug-in file {path!r} cannot be imported")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            del sys.modules[name]
            raise PluginError(f"plug-in file {path!r} failed to import: {exc!r}") from exc
        return module
    try:
        return importlib.import_module(path)
    except ImportError as exc:
        raise PluginError(f"plug-in module {path!r} cannot be imported: {exc}") from exc
    except Exception as exc:
        raise PluginError(f"plug-in module {path!r} failed to import: {exc!r}") from exc


def _digest(module: ModuleType) -> str:
    try:
        source = inspect.getsourcefile(module)
        data = Path(source).read_bytes() if source else b""
    except (OSError, TypeError):
        data = b""
    return hashlib.sha256(data).hexdigest()[:8]


def load_plugin(path: str) -> LoadedPlugin:
    """Import ``package.module:attr`` or ``file.py:attr`` and check it is an ``AlphaPlugin``."""
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
    return LoadedPlugin(obj, path.strip(), _digest(module))


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


def plugins() -> dict[str, LoadedPlugin]:
    """Every configured plug-in by name, loaded once per configuration."""
    paths = configured_paths()
    if paths not in _cache:
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
    return _cache[paths]


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
