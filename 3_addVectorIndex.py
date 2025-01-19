# from langchain_neo4j import Neo4jGraph
# from langchain_community.vectorstores import Neo4jVector
# from langchain_ollama import OllamaEmbeddings

# graph = Neo4jGraph(url="bolt://localhost:7687",
#                    username="neo4j", password="neo4jneo4j")

# # Add vecotors to entities for vector search
# # https://stackoverflow.com/questions/78173243/vector-store-created-using-existing-graph-for-multiple-nodes-labels
# Neo4jVector.from_existing_graph(
#     OllamaEmbeddings(model="mxbai-embed-large"),
#     # search_type="hybrid",
#     node_label="Document",
#     text_node_properties=["text"],
#     embedding_node_property="embedding",
#     url="bolt://localhost:7687",
#     username="neo4j",
#     password="neo4jneo4j",
#     index_name="documents",
#     keyword_index_name="text_index",
# )