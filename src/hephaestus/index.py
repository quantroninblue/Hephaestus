"""Deterministic project indexing and reference graph helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from hephaestus.model import EngineeringProject, Reference


@dataclass(frozen=True, slots=True)
class IndexedItem:
    kind: str
    id: str
    collection: str
    index: int
    value: object

    @property
    def path(self) -> str:
        return f"{self.collection}[{self.index}]"


@dataclass(frozen=True, slots=True)
class ReferenceEdge:
    source_path: str
    target: Reference


@dataclass(frozen=True, slots=True)
class ProjectIndex:
    project: EngineeringProject
    items: dict[tuple[str, str], IndexedItem]
    global_ids: dict[str, tuple[IndexedItem, ...]]
    incoming: dict[tuple[str, str], tuple[ReferenceEdge, ...]]
    outgoing: dict[str, tuple[ReferenceEdge, ...]]
    component_items: dict[str, tuple[IndexedItem, ...]]

    @classmethod
    def build(cls, project: EngineeringProject) -> ProjectIndex:
        items: dict[tuple[str, str], IndexedItem] = {}
        global_map: dict[str, list[IndexedItem]] = defaultdict(list)
        component_map: dict[str, list[IndexedItem]] = defaultdict(list)
        for collection_name, kind in _COLLECTION_KINDS.items():
            for index, value in enumerate(getattr(project, collection_name)):
                indexed = IndexedItem(kind, value.id, collection_name, index, value)
                items[(kind, value.id)] = indexed
                global_map[value.id].append(indexed)
                if collection_name in _COMPONENT_COLLECTIONS:
                    component_map[value.id].append(indexed)

        incoming: dict[tuple[str, str], list[ReferenceEdge]] = defaultdict(list)
        outgoing: dict[str, list[ReferenceEdge]] = defaultdict(list)
        for collection_name in EngineeringProject.COLLECTIONS:
            for index, value in enumerate(getattr(project, collection_name)):
                root_path = f"{collection_name}[{index}]"
                for path, ref in iter_references_in_value(value, root_path):
                    edge = ReferenceEdge(path, ref)
                    incoming[(ref.kind, ref.id)].append(edge)
                    outgoing[root_path].append(edge)
                    if ref.kind == "component":
                        incoming[("component", ref.id)].append(edge)
        return cls(
            project=project,
            items=items,
            global_ids={key: tuple(value) for key, value in global_map.items()},
            incoming={key: tuple(value) for key, value in incoming.items()},
            outgoing={key: tuple(value) for key, value in outgoing.items()},
            component_items={key: tuple(value) for key, value in component_map.items()},
        )

    def resolve(self, ref: Reference) -> IndexedItem | None:
        if ref.kind == "project":
            return (
                IndexedItem("project", self.project.id, "project", 0, self.project)
                if ref.id == self.project.id
                else None
            )
        if ref.kind == "component":
            matches = self.component_items.get(ref.id, ())
            return matches[0] if len(matches) == 1 else None
        return self.items.get((ref.kind, ref.id))

    def resolve_component_as(
        self, ref: Reference, allowed_collections: set[str]
    ) -> IndexedItem | None:
        if ref.kind != "component":
            return None
        matches = tuple(
            item
            for item in self.component_items.get(ref.id, ())
            if item.collection in allowed_collections
        )
        return matches[0] if len(matches) == 1 else None

    def incoming_references(self, kind: str, id: str) -> tuple[ReferenceEdge, ...]:
        return self.incoming.get((kind, id), ())

    def outgoing_references(self, root_path: str) -> tuple[ReferenceEdge, ...]:
        return self.outgoing.get(root_path, ())

    def removal_blockers(self, kind: str, id: str) -> tuple[str, ...]:
        return tuple(edge.source_path for edge in self.incoming_references(kind, id))

    def component_ambiguities(self) -> dict[str, tuple[IndexedItem, ...]]:
        return {key: value for key, value in self.component_items.items() if len(value) > 1}


_COLLECTION_KINDS = {
    "requirements": "requirement",
    "assumptions": "assumption",
    "entities": "entity",
    "frames": "frame",
    "transforms": "transform",
    "materials": "material",
    "geometries": "geometry",
    "bodies": "body",
    "joints": "joint",
    "connections": "connection",
    "actuators": "actuator",
    "sensors": "sensor",
    "electrical_devices": "electrical_device",
    "electrical_ports": "electrical_port",
    "power_connections": "connection",
    "communication_buses": "bus",
    "communication_interfaces": "communication_interface",
    "compute_devices": "compute_device",
    "software_components": "software_component",
    "simulation_scenarios": "scenario",
    "engineering_decisions": "decision",
}
_COMPONENT_COLLECTIONS = {"materials", "geometries", "communication_interfaces"}


def iter_references_in_value(value: Any, path: str) -> Iterable[tuple[str, Reference]]:
    if isinstance(value, Reference):
        yield path, value
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from iter_references_in_value(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_references_in_value(item, f"{path}.{key}")
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            if name == "COLLECTIONS":
                continue
            yield from iter_references_in_value(getattr(value, name), f"{path}.{name}")
