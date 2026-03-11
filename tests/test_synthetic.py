from cypher_slm.synthetic import build_demo_schemas, generate_synthetic_examples


def test_generate_synthetic_examples_returns_examples_for_each_schema():
    schemas = build_demo_schemas()
    examples = generate_synthetic_examples(schemas)
    schema_ids = {example.schema_id for example in examples}
    assert schema_ids == {"movies", "social", "commerce"}
    assert len(examples) >= 18
    assert all("LIMIT" in example.cypher or "count(" in example.cypher.lower() for example in examples)
