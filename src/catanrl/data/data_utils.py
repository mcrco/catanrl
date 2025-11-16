def is_non_graph_feature(col: str) -> bool:
    if not col.startswith('F_'):
        return False
    return not (
        col.startswith('F_NODE')
        or col.startswith('F_EDGE')
        or col.startswith('F_TILE')
        or col.startswith('F_PORT')
    )