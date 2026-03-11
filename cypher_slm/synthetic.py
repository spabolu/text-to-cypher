from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from neo4j import GraphDatabase

from .data import CypherExample, infer_difficulty, normalize_cypher


@dataclass(slots=True)
class NodeType:
    label: str
    properties: dict[str, str]


@dataclass(slots=True)
class RelationshipType:
    rel_type: str
    start: str
    end: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GraphSchema:
    schema_id: str
    description: str
    nodes: list[NodeType]
    relationships: list[RelationshipType]

    def to_schema_text(self) -> str:
        node_lines = []
        for node in self.nodes:
            properties = ", ".join(f"{name}: {kind}" for name, kind in node.properties.items()) or "no properties"
            node_lines.append(f"Node {node.label} {{ {properties} }}")
        rel_lines = []
        for rel in self.relationships:
            properties = ", ".join(f"{name}: {kind}" for name, kind in rel.properties.items()) or "no properties"
            rel_lines.append(f"Relationship (:{rel.start})-[:{rel.rel_type} {{ {properties} }}]->(:{rel.end})")
        return "\n".join([self.description, *node_lines, *rel_lines])


def build_demo_schemas() -> list[GraphSchema]:
    return [
        GraphSchema(
            schema_id="movies",
            description="A movie recommendation graph with people, movies, genres, and ratings.",
            nodes=[
                NodeType("Person", {"name": "string", "born": "integer"}),
                NodeType("Movie", {"title": "string", "released": "integer", "tagline": "string"}),
                NodeType("Genre", {"name": "string"}),
            ],
            relationships=[
                RelationshipType("ACTED_IN", "Person", "Movie", {"roles": "list[string]"}),
                RelationshipType("DIRECTED", "Person", "Movie"),
                RelationshipType("IN_GENRE", "Movie", "Genre"),
            ],
        ),
        GraphSchema(
            schema_id="social",
            description="A social network graph tracking users, posts, and topics.",
            nodes=[
                NodeType("User", {"username": "string", "country": "string", "followers": "integer"}),
                NodeType("Post", {"id": "string", "likes": "integer", "created_at": "datetime"}),
                NodeType("Topic", {"name": "string"}),
            ],
            relationships=[
                RelationshipType("POSTED", "User", "Post"),
                RelationshipType("FOLLOWS", "User", "User", {"since": "date"}),
                RelationshipType("TAGGED", "Post", "Topic"),
            ],
        ),
        GraphSchema(
            schema_id="commerce",
            description="An ecommerce graph with customers, orders, products, and categories.",
            nodes=[
                NodeType("Customer", {"name": "string", "segment": "string", "country": "string"}),
                NodeType("Order", {"id": "string", "total": "float", "ordered_at": "datetime"}),
                NodeType("Product", {"name": "string", "price": "float"}),
                NodeType("Category", {"name": "string"}),
            ],
            relationships=[
                RelationshipType("PLACED", "Customer", "Order"),
                RelationshipType("CONTAINS", "Order", "Product", {"quantity": "integer"}),
                RelationshipType("IN_CATEGORY", "Product", "Category"),
            ],
        ),
    ]


def _first_property(node: NodeType) -> tuple[str, str] | None:
    return next(iter(node.properties.items()), None)


def _node_lookup(schema: GraphSchema, label: str) -> NodeType:
    for node in schema.nodes:
        if node.label == label:
            return node
    raise KeyError(label)


def _format_property_types(value) -> str:
    if isinstance(value, list):
        return "/".join(str(item) for item in value) or "unknown"
    if value in (None, ""):
        return "unknown"
    return str(value)


def _parse_label(raw) -> str:
    if isinstance(raw, list):
        return str(raw[0]) if raw else "Unknown"
    text = str(raw).strip()
    if text.startswith(":"):
        text = text[1:]
    return text or "Unknown"


def _safe_label(label: str) -> str:
    return label.replace("`", "``")


def introspect_neo4j_schema(
    uri: str,
    username: str,
    password: str,
    database: str | None = None,
    schema_id: str = "neo4j_live",
) -> GraphSchema:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            node_rows = session.run(
                "CALL db.schema.nodeTypeProperties() "
                "YIELD nodeLabels, propertyName, propertyTypes "
                "RETURN nodeLabels, propertyName, propertyTypes"
            )
            node_map: dict[str, dict[str, str]] = {}
            for record in node_rows:
                label = _parse_label(record.get("nodeLabels"))
                node_map.setdefault(label, {})
                property_name = record.get("propertyName")
                if property_name:
                    node_map[label][str(property_name)] = _format_property_types(record.get("propertyTypes"))

            rel_rows = session.run(
                "CALL db.schema.relTypeProperties() "
                "YIELD relType, sourceNodeLabels, targetNodeLabels, propertyName, propertyTypes "
                "RETURN relType, sourceNodeLabels, targetNodeLabels, propertyName, propertyTypes"
            )
            rel_map: dict[tuple[str, str, str], dict[str, str]] = {}
            for record in rel_rows:
                rel_type = str(record.get("relType") or "").strip(":") or "RELATED_TO"
                start = _parse_label(record.get("sourceNodeLabels"))
                end = _parse_label(record.get("targetNodeLabels"))
                key = (rel_type, start, end)
                rel_map.setdefault(key, {})
                property_name = record.get("propertyName")
                if property_name:
                    rel_map[key][str(property_name)] = _format_property_types(record.get("propertyTypes"))

        nodes = [NodeType(label=label, properties=properties) for label, properties in sorted(node_map.items())]
        relationships = [
            RelationshipType(rel_type=rel_type, start=start, end=end, properties=properties)
            for (rel_type, start, end), properties in sorted(rel_map.items())
        ]
        if not nodes:
            raise RuntimeError("Neo4j schema introspection returned no node labels. The database may be empty.")
        return GraphSchema(
            schema_id=schema_id,
            description="A live Neo4j Aura schema introspected from database metadata.",
            nodes=nodes,
            relationships=relationships,
        )
    finally:
        driver.close()


def generate_synthetic_examples(
    schemas: Iterable[GraphSchema],
    property_list_limit: int | None = 10,
    relationship_list_limit: int | None = 10,
    top_k: int = 5,
) -> list[CypherExample]:
    examples: list[CypherExample] = []
    for schema in schemas:
        schema_text = schema.to_schema_text()
        for node in schema.nodes:
            first_property = _first_property(node)
            if first_property is not None:
                property_name, _ = first_property
                list_query = f"MATCH (n:{_safe_label(node.label)}) RETURN n.{property_name} AS {property_name}"
                if property_list_limit is not None:
                    list_query += f" LIMIT {property_list_limit}"
                examples.append(
                    CypherExample(
                        schema_id=schema.schema_id,
                        schema_text=schema_text,
                        question=f"List up to {property_list_limit} {property_name} values for {node.label.lower()} nodes.",
                        cypher=normalize_cypher(list_query),
                        source="synthetic/template:list-properties",
                        difficulty="easy",
                    )
                )
            examples.append(
                CypherExample(
                    schema_id=schema.schema_id,
                    schema_text=schema_text,
                    question=f"How many {node.label.lower()} nodes are in the graph?",
                    cypher=normalize_cypher(
                        f"MATCH (n:{_safe_label(node.label)}) RETURN count(n) AS {node.label.lower()}Count"
                    ),
                    source="synthetic/template:count-nodes",
                    difficulty="easy",
                )
            )
        for rel in schema.relationships:
            start_node = _node_lookup(schema, rel.start)
            end_node = _node_lookup(schema, rel.end)
            start_property_info = _first_property(start_node)
            end_property_info = _first_property(end_node)
            if start_property_info and end_property_info:
                start_property, _ = start_property_info
                end_property, _ = end_property_info
                rel_query = (
                    f"MATCH (a:{_safe_label(rel.start)})-[:{rel.rel_type}]->(b:{_safe_label(rel.end)}) "
                    f"RETURN a.{start_property} AS {start_property}, b.{end_property} AS {end_property}"
                )
                if relationship_list_limit is not None:
                    rel_query += f" LIMIT {relationship_list_limit}"
                templates = [
                    (
                        f"Show up to {relationship_list_limit} {start_node.label.lower()} to {end_node.label.lower()} {rel.rel_type.lower()} connections.",
                        rel_query,
                    ),
                    (
                        f"Find the top {top_k} {start_node.label.lower()} nodes by number of {rel.rel_type} relationships.",
                        f"MATCH (a:{_safe_label(rel.start)})-[:{rel.rel_type}]->(:{_safe_label(rel.end)}) RETURN a.{start_property} AS {start_property}, count(*) AS relationshipCount ORDER BY relationshipCount DESC LIMIT {top_k}",
                    ),
                ]
            else:
                templates = [
                    (
                        f"How many {rel.rel_type} relationships exist between {start_node.label.lower()} and {end_node.label.lower()} nodes?",
                        f"MATCH (:{_safe_label(rel.start)})-[:{rel.rel_type}]->(:{_safe_label(rel.end)}) RETURN count(*) AS relationshipCount",
                    )
                ]
            for question, cypher in templates:
                examples.append(
                    CypherExample(
                        schema_id=schema.schema_id,
                        schema_text=schema_text,
                        question=question,
                        cypher=normalize_cypher(cypher),
                        source=f"synthetic/template:{rel.rel_type.lower()}",
                        difficulty=infer_difficulty(cypher),
                        split="test",
                    )
                )
    return examples
