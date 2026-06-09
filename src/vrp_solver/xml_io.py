from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .model import (
    Customer,
    Driver,
    Instance,
    Operation,
    Order,
    Shift,
    Solution,
    Source,
    TimeWindow,
    Trailer,
)


def _text(element: ET.Element, tag: str, default: str | None = None) -> str:
    child = element.find(tag)
    if child is None or child.text is None:
        if default is None:
            raise ValueError(f"Missing <{tag}> under <{element.tag}>")
        return default
    return child.text.strip()


def _int(element: ET.Element, tag: str, default: int | None = None) -> int:
    text_default = None if default is None else str(default)
    return int(_text(element, tag, text_default))


def _float(element: ET.Element, tag: str, default: float | None = None) -> float:
    text_default = None if default is None else str(default)
    return float(_text(element, tag, text_default))


def _int_array(element: ET.Element | None) -> tuple[int, ...]:
    if element is None:
        return ()
    if element.text and element.text.strip():
        return (int(element.text.strip()),)
    return tuple(int(child.text.strip()) for child in element if child.text)


def _float_array(element: ET.Element | None) -> tuple[float, ...]:
    if element is None:
        return ()
    return tuple(float(child.text.strip()) for child in element if child.text)


def _time_windows(element: ET.Element | None) -> tuple[TimeWindow, ...]:
    if element is None:
        return ()
    windows = []
    for child in element:
        start = _int(child, "start", _int(child, "Start", 0))
        end = _int(child, "end", _int(child, "End", 0))
        windows.append(TimeWindow(start=start, end=end))
    return tuple(windows)


def _orders(element: ET.Element | None) -> tuple[Order, ...]:
    if element is None:
        return ()
    orders = []
    for child in element:
        quantity = _float(child, "Quantity", _float(child, "quantity", 0.0))
        orders.append(
            Order(
                quantity=quantity,
                earliest_time=_int(child, "earliestTime", 0),
                latest_time=_int(child, "latestTime", 0),
                quantity_flexibility=_int(child, "orderQuantityFlexibility", 100),
            )
        )
    return tuple(orders)


def _matrix_int(element: ET.Element) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value.text.strip()) for value in row if value.text) for row in element)


def _matrix_float(element: ET.Element) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value.text.strip()) for value in row if value.text) for row in element)


def load_instance(path: str | Path) -> Instance:
    root = ET.parse(path).getroot()
    horizon = _int(root, "horizon")
    unit = _int(root, "unit")
    drivers = tuple(_driver(child) for child in root.find("drivers") or [])
    trailers = tuple(_trailer(child) for child in root.find("trailers") or [])
    all_trailer_ids = tuple(trailer.index for trailer in trailers)
    sources = tuple(
        _source(child, all_trailer_ids)
        for child in root.find("sources") or []
    )
    customers = tuple(
        _customer(child, horizon_minutes=horizon * unit)
        for child in root.find("customers") or []
    )

    bases = root.find("bases")
    if bases is None:
        raise ValueError("Missing <bases>")

    return Instance(
        name=_text(root, "name", Path(path).stem),
        unit=unit,
        horizon=horizon,
        time_matrix=_matrix_int(root.find("timeMatrices") or ET.Element("empty")),
        distance_matrix=_matrix_float(root.find("DistMatrices") or ET.Element("empty")),
        base_index=_int(bases, "index"),
        drivers=drivers,
        trailers=trailers,
        sources=sources,
        customers=customers,
    )


def _driver(element: ET.Element) -> Driver:
    return Driver(
        index=_int(element, "index"),
        min_inter_shift_duration=_int(element, "minInterSHIFTDURATION"),
        max_driving_duration=_int(element, "maxDrivingDuration"),
        trailer_ids=_int_array(element.find("trailer")),
        time_windows=_time_windows(element.find("timewindows")),
        layover_duration=_int(element, "LayoverDuration", 900),
        time_cost=_float(element, "TimeCost", 0.0),
        layover_cost=_float(element, "LayoverCost", 0.0),
    )


def _trailer(element: ET.Element) -> Trailer:
    return Trailer(
        index=_int(element, "index"),
        capacity=_float(element, "Capacity"),
        initial_quantity=_float(element, "InitialQuantity"),
        distance_cost=_float(element, "DistanceCost"),
    )


def _source(element: ET.Element, all_trailer_ids: tuple[int, ...]) -> Source:
    allowed_trailers = _int_array(element.find("allowedTrailers"))
    return Source(
        index=_int(element, "index"),
        allowed_trailers=allowed_trailers or all_trailer_ids,
        setup_time=_int(element, "setupTime"),
    )


def _customer(element: ET.Element, *, horizon_minutes: int) -> Customer:
    time_windows = _time_windows(element.find("timewindows"))
    return Customer(
        index=_int(element, "index"),
        layover_customer=bool(_int(element, "LayoverCustomer", 0)),
        call_in=bool(_int(element, "callIn", 0)),
        orders=_orders(element.find("orders")),
        setup_time=_int(element, "setupTime"),
        time_windows=time_windows or (TimeWindow(start=0, end=horizon_minutes - 1),),
        allowed_trailers=_int_array(element.find("allowedTrailers")),
        forecast=_float_array(element.find("Forecast")),
        capacity=_float(element, "Capacity", 0.0),
        initial_tank_quantity=_float(element, "InitialTankQuantity", 0.0),
        min_operation_quantity=_float(element, "MinOperationQuantity", 0.0),
        safety_level=_float(element, "SafetyLevel", 0.0),
    )


def load_solution(path: str | Path) -> Solution:
    root = ET.parse(path).getroot()
    shifts_element = root.find("Shifts")
    shifts = tuple(_shift(child) for child in shifts_element or [])
    return Solution(shifts=shifts)


def _shift(element: ET.Element) -> Shift:
    operations_element = element.find("operations")
    operations = tuple(_operation(child) for child in operations_element or [])
    return Shift(
        index=_int(element, "index"),
        driver=_int(element, "driver"),
        trailer=_int(element, "trailer"),
        start=_int(element, "start"),
        operations=operations,
    )


def _operation(element: ET.Element) -> Operation:
    return Operation(
        point=_int(element, "point"),
        arrival=_int(element, "arrival"),
        quantity=_float(element, "Quantity"),
    )


def save_solution(solution: Solution, path: str | Path) -> None:
    root = ET.Element("IRP_Roadef_Challenge_Output")
    shifts_element = ET.SubElement(root, "Shifts")

    for output_index, shift in enumerate(
        sorted(solution.shifts, key=lambda item: (item.start, item.index))
    ):
        shift_element = ET.SubElement(shifts_element, "IRP_Roadef_Challenge_Shift_")
        ET.SubElement(shift_element, "index").text = str(output_index)
        ET.SubElement(shift_element, "driver").text = str(shift.driver)
        ET.SubElement(shift_element, "trailer").text = str(shift.trailer)
        ET.SubElement(shift_element, "start").text = str(shift.start)
        operations_element = ET.SubElement(shift_element, "operations")

        for operation in shift.operations:
            operation_element = ET.SubElement(
                operations_element,
                "IRP_Roadef_Challenge_Operation_",
            )
            ET.SubElement(operation_element, "point").text = str(operation.point)
            ET.SubElement(operation_element, "arrival").text = str(operation.arrival)
            ET.SubElement(operation_element, "Quantity").text = format(
                operation.quantity,
                ".15g",
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
