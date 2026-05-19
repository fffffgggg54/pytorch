# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict

import torch
from torch.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _annotate_noncontiguous_module_scopes(orig_graph: torch.fx.Graph) -> bool:
    """Annotate non-contiguous re-entries of the same module scope with
    ``@N`` suffixes in ``nn_module_stack`` keys.

    When a module (e.g., ``rotary_emb``) is called once, leaves scope, and then
    is called again later in the graph, ``_ModuleFrame`` would create a single
    ``InterpreterModule`` whose graph gets overwritten by the second call.
    By appending ``@N`` to the ``nn_module_stack`` key for re-entries,
    ``_ModuleFrame`` treats each call as a distinct module invocation and
    creates separate ``InterpreterModule`` instances with correct signatures.

    ``_ModuleFrame`` shares ``_modules`` between calls with the same
    ``module_id`` (the ``@``-stripped key), so parameters, buffers, and child
    submodules are not duplicated.

    Returns True if any annotations were made.
    """
    # State machine tracking scope occupancy at each depth level.
    # When the scope at depth `d` changes from key A to key B,
    # all depths >= d are marked as "left". When A re-appears,
    # we bump its call counter so it gets an @N suffix.
    active: dict[tuple[int, str], bool] = {}
    counts: dict[tuple[int, str], int] = defaultdict(int)
    prev_at_depth: dict[int, str] = {}
    modified = False

    for node in orig_graph.nodes:
        nn_module_stack = node.meta.get("nn_module_stack")
        if not nn_module_stack:
            continue

        items = list(nn_module_stack.items())
        cur_max_depth = len(items) - 1

        # Process each depth level: detect exits and re-entries
        for depth, (key, _value) in enumerate(items):
            base_key = key.split("@")[0]
            scope_id = (depth, base_key)
            prev_key = prev_at_depth.get(depth)

            # Detect scope exit: if the key at this depth changed,
            # mark the old key (and everything deeper) as inactive
            if prev_key is not None and prev_key != base_key:
                for d in list(prev_at_depth.keys()):
                    if d >= depth:
                        old_base = prev_at_depth[d]
                        active[(d, old_base)] = False

            # Detect entry/re-entry
            if not active.get(scope_id, False):
                counts[scope_id] += 1
                active[scope_id] = True

            prev_at_depth[depth] = base_key

        # Depths beyond this node's stack are no longer active
        for d in list(prev_at_depth.keys()):
            if d > cur_max_depth:
                old_base = prev_at_depth.pop(d)
                active[(d, old_base)] = False

        # Annotate keys whose scope was re-entered (count > 1)
        new_stack = {}
        needs_update = False
        for depth, (key, value) in enumerate(items):
            base_key = key.split("@")[0]
            call_num = counts.get((depth, base_key), 1)
            if call_num > 1 and "@" not in key:
                new_stack[f"{key}@{call_num - 1}"] = value
                needs_update = True
            else:
                new_stack[key] = value

        if needs_update:
            node.meta["nn_module_stack"] = new_stack
            modified = True

    return modified


def _copy_graph_attrs_to_submodules(
    orig_graph: torch.fx.Graph, new_module: torch.fx.GraphModule
):
    """Copy ``get_attr`` targets (e.g., HOP callable submodules) from the
    original graph's owning module to outlined submodules that reference them.

    ``_ModuleFrame`` copies ``get_attr`` nodes into child graphs but does not
    copy the actual attribute objects. For higher-order-op patterns like
    ``wrap_with_set_grad_enabled(False, submod_1, ...)``, ``submod_1`` must
    exist as an attribute on the ``InterpreterModule`` that contains the
    ``get_attr`` node.
    """
    src_module = orig_graph.owning_module
    if src_module is None:
        return

    for _fqn, mod in new_module.named_modules():
        if not hasattr(mod, "graph"):
            continue
        for node in mod.graph.nodes:
            if node.op != "get_attr":
                continue
            target = node.target
            if hasattr(mod, target):
                continue
            # Walk dotted path on source module
            obj = src_module
            try:
                for part in target.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            # Set on the destination module using nested path
            parts = target.split(".")
            dest = mod
            for part in parts[:-1]:
                if not hasattr(dest, part):
                    setattr(dest, part, torch.nn.Module())
                dest = getattr(dest, part)
            setattr(dest, parts[-1], obj)


def _outline_submodules(orig_graph: torch.fx.Graph) -> torch.fx.GraphModule:
    # Pre-process: annotate non-contiguous re-uses of the same module scope
    # so that _ModuleFrame creates separate InterpreterModules per call.
    _annotate_noncontiguous_module_scopes(orig_graph)

    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()

    # Post-process: copy HOP get_attr targets to outlined submodules
    _copy_graph_attrs_to_submodules(orig_graph, new_module)

    new_module.graph.lint()
    new_module.recompile()
    return new_module
