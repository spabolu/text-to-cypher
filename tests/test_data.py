from cypher_slm.data import CypherExample, build_training_corpus, normalize_cypher


def test_normalize_cypher_removes_trailing_semicolon_and_extra_spaces():
    raw = " MATCH (n:Movie)   RETURN n.title AS title ; \n"
    assert normalize_cypher(raw) == "MATCH (n:Movie) RETURN n.title AS title"


def test_build_training_corpus_assigns_held_out_schema_to_test():
    public = [
        CypherExample(
            schema_id="movies",
            schema_text="movie schema",
            question="How many movies are there?",
            cypher="MATCH (m:Movie) RETURN count(m) AS movieCount",
            source="public",
        )
    ]
    synthetic = [
        CypherExample(
            schema_id="commerce",
            schema_text="commerce schema",
            question="How many orders are there?",
            cypher="MATCH (o:Order) RETURN count(o) AS orderCount",
            source="synthetic",
        )
    ]
    corpus = build_training_corpus(public, synthetic, held_out_schema_ids=["commerce"])
    split_by_schema = {row.schema_id: row.split for row in corpus}
    assert split_by_schema["commerce"] == "test"
    assert split_by_schema["movies"] in {"train", "validation"}
