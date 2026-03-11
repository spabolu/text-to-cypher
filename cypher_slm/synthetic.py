from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

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
            properties = ", ".join(f"{name}: {kind}" for name, kind in node.properties.items())
            node_lines.append(f"Node {node.label} {{ {properties} }}")
        rel_lines = []
        for rel in self.relationships:
            properties = ", ".join(f"{name}: {kind}" for name, kind in rel.properties.items()) or "no properties"
            rel_lines.append(f"Relationship (: {rel.start})-[:{rel.rel_type} {{ {properties} }}]->(: {rel.end})")
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


def _first_property(node: NodeType) -> tuple[str, str]:
    return next(iter(node.properties.items()))


def _node_lookup(schema: GraphSchema, label: str) -> NodeType:
    for node in schema.nodes:
        if node.label == label:
            return node
    raise KeyError(label)


def generate_synthetic_examples(schemas: Iterable[GraphSchema]) -> list[CypherExample]:
    examples: list[CypherExample] = []
    for schema in schemas:
        schema_text = schema.to_schema_text()
        for node in schema.nodes:
            property_name, _ = _first_property(node)
            examples.extend(
                [
                    CypherExample(
                        schema_id=schema.schema_id,
                        schema_text=schema_text,
                        question=f"List the {property_name} values for all {node.label.lower()} nodes.",
                        cypher=normalize_cypher(f"MATCH (n:{node.label}) RETURN n.{property_name} AS {property_name}"),
                        source="synthetic/template:list-properties",
                        difficulty="easy",
                    ),
                    CypherExample(
                        schema_id=schema.schema_id,
                        schema_text=schema_text,
                        question=f"How many {node.label.lower()} nodes are in the graph?",
                        cypher=normalize_cypher(f"MATCH (n:{node.label}) RETURN count(n) AS {node.label.lower()}Count"),
                        source="synthetic/template:count-nodes",
                        difficulty="easy",
                    ),
                ]
            )
        for rel in schema.relationships:
            start_node = _node_lookup(schema, rel.start)
            end_node = _node_lookup(schema, rel.end)
            start_property, _ = _first_property(start_node)
            end_property, _ = _first_property(end_node)
            templates = [
                (
                    f"Show each {start_node.label.lower()} and the {end_property} of the {end_node.label.lower()} nodes connected by {rel.rel_type}.",
                    f"MATCH (a:{rel.start})-[:{rel.rel_type}]->(b:{rel.end}) RETURN a.{start_property} AS {start_property}, b.{end_property} AS {end_property}",
                ),
                (
                    f"Find the top 5 {start_node.label.lower()} nodes by number of {rel.rel_type} relationships.",
                    f"MATCH (a:{rel.start})-[:{rel.rel_type}]->(:{rel.end}) RETURN a.{start_property} AS {start_property}, count(*) AS relationshipCount ORDER BY relationshipCount DESC LIMIT 5",
                ),
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
                    )
                )
    return examples
