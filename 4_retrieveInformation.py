from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

graph = Neo4jGraph(url="bolt://localhost:7687",
                   username="neo4j", password="neo4jneo4j")

llm = OllamaFunctions(
    model="llama3.1", temperature=0, format="json")

# Add vecotors to entities for vector search
docuement_index = Neo4jVector.from_existing_graph(
    OllamaEmbeddings(model="mxbai-embed-large"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jneo4j"
)
document_retriever = docuement_index.as_retriever()
# Add vecotors to entities for vector search
person_index = Neo4jVector.from_existing_graph(
    OllamaEmbeddings(model="mxbai-embed-large"),
    search_type="hybrid",
    node_label="Person",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jneo4j"
)
person_retriever = person_index.as_retriever()

# Retrieve with embeddings

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting the meaing of the text",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = llm.with_structured_output(Entities)

# res = entity_chain.invoke("What happened in Stockholm")
# print(res.names) # names=['Stockholm']

# Fulltext index query
def graph_retriever(question):
    result = ""
    entities = entity_chain.invoke(question)
    print(entities)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node, score
            CALL (node, score){
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 500
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# print(graph_retriever("Pierre"))

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

def full_retriever(question: str): #Hybrid between graph and vector data
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in document_retriever.invoke(question)]
    person_data = [el.page_content for el in person_retriever.invoke(question)]
    
    final_data = f"""Graph data: {graph_data}
    vector data: {"#Document ".join(vector_data)}
    person data: {"#Person ".join(person_data)}
    """
    print(f"############################### \n {final_data} \n############################### ")
    return final_data

template = """Answer the question based only on the following context: {context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {
        "context": full_retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke(input="What did Paul do"))
