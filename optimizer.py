# Got some late night help
# https://chatgpt.com/c/69dabd86-1944-8391-b3f4-f1381d37e717
from dataclasses import dataclass
from collections import defaultdict
from typing import Mapping, Iterable, Any

import rg_database_interactions as rgdb


@dataclass(frozen=True)
class Option:
    price: float
    items: tuple[tuple[int, int], ...]          # items that must still be bought / provided
    path: tuple[Any, ...]                       # nodes used for this plan
    inventory_after: tuple[tuple[int, int], ...]  # frozen remaining inventory


def freeze_inventory(inv: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """
    This function turns a dict of inventories into a cachable tuple.
    :param inv:
    :return:
    """
    return tuple(sorted((item_id, amount) for item_id, amount in inv.items() if amount > 0))


def consume_from_inventory(
    item_id: int,
    required_amount: int,
    inventory_key: tuple[tuple[int, int], ...],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """
    Pure version of compute_necessary_amount:
    consumes as much as possible from the available inventory and returns
    (amount_still_needed, new_inventory_key).
    """
    inv = dict(inventory_key)

    available = inv.get(item_id, 0)
    used = min(available, required_amount)
    still_needed = required_amount - used

    if used:
        remaining = available - used
        if remaining > 0:
            inv[item_id] = remaining
        else:
            inv.pop(item_id, None)

    return still_needed, freeze_inventory(inv)


def merge_item_lists(*item_lists: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """
    This turns an arbitrary number of tuples for available items into one tuple.
    :param item_lists:
    :return:
    """
    merged = defaultdict(int)
    for item_list in item_lists:
        for item_id, amount in item_list:
            if amount:
                merged[item_id] += amount
    return freeze_inventory(merged)


def top_k_distinct(options: list[Option], k: int) -> list[Option]:
    """
    Keep only distinct plans and return the k cheapest.
    Distinctness is defined by:
      - purchased items
      - node path
      - resulting remaining inventory
    """
    best_by_key: dict[tuple, Option] = {}

    for opt in options:
        key = (
            opt.items,
            tuple(id(node) for node in opt.path),
            opt.inventory_after,
        )
        prev = best_by_key.get(key)
        if prev is None or opt.price < prev.price:
            best_by_key[key] = opt

    return sorted(
        best_by_key.values(),
        key=lambda o: (o.price, len(o.path), o.items)
    )[:k]


def k_best_crafting_paths(
    root,
    recent_prices: Mapping[int, float],
    available_items: Mapping[int, int] | None = None,
    k: int = 1,
):
    """
    Returns up to k cheapest crafting plans.

    Each returned entry is:
        (total_price, [(item_id, amount_needed)], [path_nodes], {remaining_inventory})

    Assumptions:
    - The graph is a DAG.
    - An ItemNode child alternative crafts the whole required amount for that node,
      exactly like in your current implementation.
    """
    if available_items is None:
        available_items = {}

    memo: dict[tuple[int, tuple[tuple[int, int], ...]], list[Option]] = {}

    def dfs(node, inventory_key: tuple[tuple[int, int], ...]) -> list[Option]:

        # if the already reached this node with the same available items, we can use our cache
        cache_key = (id(node), inventory_key)
        if cache_key in memo:
            return memo[cache_key]

        # Leaf item: consume from available inventory, buy the rest.
        if len(node.children) == 0:
            assert isinstance(node, rgdb.ItemNode), "Something is off."

            required_amount, inv_after = consume_from_inventory(
                node.id,
                node.required_amount,
                inventory_key,
            )

            result = [
                Option(
                    price=recent_prices[node.id] * required_amount,
                    items=((node.id, required_amount),) if required_amount > 0 else (),
                    path=(node,),
                    inventory_after=inv_after,
                )
            ]

        # SpellNode: AND-combine all child plans while carrying forward the inventory state.
        elif isinstance(node, rgdb.SpellNode):
            children = tuple(node.children.values())
            combine_memo: dict[tuple[int, tuple[tuple[int, int], ...]], list[Option]] = {}

            def combine_children(
                child_idx: int,
                curr_inventory_key: tuple[tuple[int, int], ...],
            ) -> list[Option]:
                key = (child_idx, curr_inventory_key)
                if key in combine_memo:
                    return combine_memo[key]

                if child_idx == len(children):
                    out = [Option(0, (), (), curr_inventory_key)]
                else:
                    out: list[Option] = []
                    child = children[child_idx]

                    left_options = dfs(child, curr_inventory_key)
                    for left in left_options:
                        right_options = combine_children(child_idx + 1, left.inventory_after)
                        for right in right_options:
                            out.append(
                                Option(
                                    price=left.price + right.price,
                                    items=merge_item_lists(left.items, right.items),
                                    path=left.path + right.path,
                                    inventory_after=right.inventory_after,
                                )
                            )

                    out = top_k_distinct(out, k)

                combine_memo[key] = out
                return out

            combined = combine_children(0, inventory_key)
            result = [
                Option(
                    price=opt.price,
                    items=opt.items,
                    path=opt.path + (node,),
                    inventory_after=opt.inventory_after,
                )
                for opt in combined
            ]

        # ItemNode: OR-node, choose either direct buy/use or one of the crafting alternatives.
        elif isinstance(node, rgdb.ItemNode):
            candidates: list[Option] = []

            # Option 1: use inventory / buy the item directly
            required_amount, inv_after = consume_from_inventory(
                node.id,
                node.required_amount,
                inventory_key,
            )
            candidates.append(
                Option(
                    price=recent_prices[node.id] * required_amount,
                    items=((node.id, required_amount),) if required_amount > 0 else (),
                    path=(node,),
                    inventory_after=inv_after,
                )
            )

            # Option 2: craft via one of the child spells
            for child in node.children.values():
                candidates.extend(dfs(child, inventory_key))

            result = top_k_distinct(candidates, k)

        else:
            raise ValueError("Something is off.")

        memo[cache_key] = top_k_distinct(result, k)
        return memo[cache_key]

    final_options = dfs(root, freeze_inventory(available_items))

    return [
        (
            opt.price,
            list(opt.items),
            list(opt.path),
            dict(opt.inventory_after),
        )
        for opt in final_options
    ]