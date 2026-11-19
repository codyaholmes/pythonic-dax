"""Streamlit widgets for a pythonic-dax model: slicers, cards, tables and charts.

This is an optional add-on (``pip install pythonic-dax[streamlit]``). The engine and
the framework-neutral :mod:`pythonic_dax.report` layer don't import Streamlit, so the
same model works in any other front end.

    import pythonic_dax.streamlit as pst

    @pst.cached_model
    def load():
        ...  # build and return the Model once; reruns reuse it

    report = pst.report(load())
    pst.multiselect(report, Customer["Region"])
    pst.date_range(report, Date["Date"])
    pst.card(report, total_sales)
    pst.bar_chart(report, Date["Month"], total_sales, key="by_month")  # click to filter
    pst.table(report, [Product["Category"]], [total_sales, margin], key="by_category")

Labels use display names, help tooltips use descriptions, and values use format
strings. Clicking a bar or a table row filters the other visuals on the page, like
Power BI's cross-filtering; click it again (or clear the selection) to undo.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

try:
    import streamlit as st
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "pythonic_dax.streamlit needs Streamlit: pip install 'pythonic-dax[streamlit]'"
    ) from e

from .model import Model
from .objects import Column
from .report import Card, Report, ValueSpec, Visual

__all__ = [
    "bar_chart",
    "cached_model",
    "card",
    "date_range",
    "dropdown",
    "line_chart",
    "multiselect",
    "report",
    "table",
]

F = TypeVar("F", bound=Callable[..., Model])

_STATE = "_pdax_cross_filters"  # session state: {visual key: (table, column, values)}


def cached_model(builder: F) -> F:
    """Decorate the function that builds the model, so it runs once per server and
    every rerun and session shares the result (``st.cache_resource``)."""
    cached = st.cache_resource(show_spinner="Loading the model...")(builder)

    @wraps(builder)
    def wrapper(*args: Any, **kwargs: Any) -> Model:
        return cached(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def report(model: Model) -> Report:
    """A :class:`Report` for this run, with the cross-filter selections made by
    clicking visuals on earlier runs already applied."""
    rep = Report(model)
    for key, (table, column, values) in st.session_state.get(_STATE, {}).items():
        rep.select(model.table(table)[column], values, source=key)
    return rep


# --- slicers ----------------------------------------------------------------------------


def multiselect(
    rep: Report, column: Column, *, key: str | None = None, sidebar: bool = False
) -> list[Any]:
    """A list slicer: pick any number of values. Nothing picked means no filter."""
    where = st.sidebar if sidebar else st
    picked: list[Any] = where.multiselect(
        column.display_name,
        rep.choices(column),
        help=column.description,
        key=key or f"pdax_slicer_{column.table.name}_{column.name}",
        format_func=_blank_label,
    )
    rep.select(column, picked)
    return picked


def dropdown(
    rep: Report,
    column: Column,
    *,
    key: str | None = None,
    sidebar: bool = False,
    all_label: str = "All",
    default: Any = None,
) -> Any:
    """A dropdown slicer: one value, or ``all_label`` for no filter. ``default`` is the
    value picked when the page first opens (None for all)."""
    where = st.sidebar if sidebar else st
    choices = rep.choices(column)
    index = where.selectbox(
        column.display_name,
        range(len(choices) + 1),  # 0 is "All"; indices survive Streamlit's state handling
        index=choices.index(default) + 1 if default in choices else 0,
        help=column.description,
        key=key or f"pdax_slicer_{column.table.name}_{column.name}",
        format_func=lambda i: all_label if i == 0 else _blank_label(choices[i - 1]),
    )
    picked = None if not index else choices[index - 1]
    rep.select(column, None if not index else [picked])
    return picked


def date_range(
    rep: Report, column: Column, *, key: str | None = None, sidebar: bool = False
) -> tuple[dt.date, dt.date] | None:
    """A between slicer on a date column, from its first to its last date."""
    values = [v for v in rep.choices(column) if v is not None]
    if not values:
        return None
    first, last = min(values).date(), max(values).date()
    where = st.sidebar if sidebar else st
    picked = where.date_input(
        column.display_name,
        (first, last),
        min_value=first,
        max_value=last,
        help=column.description,
        key=key or f"pdax_slicer_{column.table.name}_{column.name}",
    )
    if not isinstance(picked, tuple) or len(picked) != 2:
        return None  # the user is still picking the end date
    start, end = picked
    if (start, end) == (first, last):
        rep.select_range(column, None, None)
    else:
        rep.select_range(column, start, end)
    return start, end


# --- visuals ----------------------------------------------------------------------------


def card(rep: Report, value: ValueSpec, *, delta: ValueSpec | None = None) -> Card:
    """A card (``st.metric``): the measure's label, formatted value and description.
    ``delta`` adds a second measure, such as growth over last year, below it."""
    result = rep.card(value)
    delta_text = None
    if delta is not None:
        delta_text = rep.card(delta).text or None
    st.metric(result.label, result.text or "(Blank)", delta=delta_text, help=result.tooltip)
    return result


def table(
    rep: Report,
    rows: Sequence[Column],
    values: Sequence[ValueSpec],
    *,
    key: str | None = None,
    totals: bool = True,
    sort_by: Column | ValueSpec | None = None,
    descending: bool = False,
    height: int | None = None,
) -> Visual:
    """A table visual with a total row. With a ``key``, clicking a row filters the
    other visuals to that row's values."""
    visual = rep.table(
        rows, values, totals=totals, name=key, sort_by=sort_by, descending=descending
    )
    config = {
        _label(visual, f.key): st.column_config.TextColumn(f.label, help=f.tooltip)
        for f in visual.fields
    }
    options: dict[str, Any] = {}
    if key is not None and rows:
        options = {
            "key": key,
            "on_select": _on_table_select(key, visual, rows[0]),
            "selection_mode": "single-row",
        }
    st.dataframe(
        visual.text,
        column_config=config,
        hide_index=True,
        use_container_width=True,
        height=height or min(35 * (visual.text.height + 1) + 3, 800),
        **options,
    )
    return visual


def bar_chart(
    rep: Report,
    category: Column,
    value: ValueSpec,
    *,
    key: str | None = None,
    horizontal: bool = False,
    sort_by: Column | ValueSpec | None = None,
    descending: bool = False,
    height: int = 300,
) -> Visual:
    """A bar chart of one measure by one column. With a ``key``, clicking a bar filters
    the other visuals to it."""
    return _chart(rep, category, value, "bar", key, horizontal, sort_by, descending, height)


def line_chart(
    rep: Report,
    category: Column,
    value: ValueSpec,
    *,
    key: str | None = None,
    height: int = 300,
) -> Visual:
    """A line chart of one measure by one column, in the column's sort order."""
    return _chart(rep, category, value, "line", key, False, None, False, height)


# --- helpers ----------------------------------------------------------------------------


def _blank_label(value: Any) -> str:
    if value is None:
        return "(Blank)"
    if isinstance(value, dt.datetime) and value.time() == dt.time():
        return value.strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
    return str(value)


def _label(visual: Visual, key: str) -> str:
    for f, name in zip(visual.fields, visual.text.columns, strict=True):
        if f.key == key:
            return name
    raise KeyError(key)


def _remember(key: str, column: Column | None, values: list[Any] | None) -> None:
    state = st.session_state.setdefault(_STATE, {})
    if column is None or not values:
        state.pop(key, None)
    else:
        state[key] = (column.table.name, column.name, values)


def _on_table_select(key: str, visual: Visual, column: Column) -> Callable[[], None]:
    """Clicking a row filters the other visuals on the row's first column, the way a
    click on a table row in Power BI does."""
    field = visual.rows[0]

    def callback() -> None:
        event = st.session_state.get(key)
        picked = list(getattr(getattr(event, "selection", None), "rows", []) or [])
        row = picked[0] if picked else None
        if row is None or row == visual.total:
            _remember(key, None, None)
        else:
            _remember(key, column, [visual.data[field.key][row]])

    return callback


def _chart(
    rep: Report,
    category: Column,
    value: ValueSpec,
    mark: str,
    key: str | None,
    horizontal: bool,
    sort_by: Column | ValueSpec | None,
    descending: bool,
    height: int,
) -> Visual:
    import altair as alt

    visual = rep.table(
        [category], [value], totals=False, name=key, sort_by=sort_by, descending=descending
    )
    cat, val = visual.fields
    labels = visual.text[visual.text.columns[0]].to_list()
    data = [
        {"category": label, "value": None if v is None else float(v), "text": t}
        for label, v, t in zip(
            labels,
            visual.data[val.key].to_list(),
            visual.text[visual.text.columns[1]].to_list(),
            strict=True,
        )
    ]
    order = labels
    cat_enc = alt.X if not horizontal else alt.Y
    val_enc = alt.Y if not horizontal else alt.X
    encodings: dict[str, Any] = {
        "x" if not horizontal else "y": cat_enc(
            "category:N",
            sort=order,
            title=cat.label,
            axis=alt.Axis(labelAngle=0 if horizontal else -45),
        ),
        "y" if not horizontal else "x": val_enc("value:Q", title=val.label),
        "tooltip": [
            alt.Tooltip("category:N", title=cat.label),
            alt.Tooltip("text:N", title=val.label),
        ],
    }
    source = alt.Data(values=data)  # type: ignore[no-untyped-call]
    base = alt.Chart(source).properties(height=height)
    chart: Any
    if key is not None and mark == "bar":
        picked = alt.selection_point(name="pick", fields=["category"])
        encodings["opacity"] = alt.condition(picked, alt.value(1.0), alt.value(0.35))
        chart = base.mark_bar().encode(**encodings).add_params(picked)
    elif mark == "bar":
        chart = base.mark_bar().encode(**encodings)
    else:
        chart = base.mark_line(point=True).encode(**encodings)

    if key is not None and mark == "bar":
        by_label = dict(zip(labels, visual.data[cat.key].to_list(), strict=True))

        def callback() -> None:
            event = st.session_state.get(key)
            points = (getattr(event, "selection", None) or {}).get("pick", []) or []
            values = [by_label[p["category"]] for p in points if p.get("category") in by_label]
            _remember(key, category, values)

        st.altair_chart(chart, use_container_width=True, key=key, on_select=callback)
    else:
        st.altair_chart(chart, use_container_width=True)
    return visual
