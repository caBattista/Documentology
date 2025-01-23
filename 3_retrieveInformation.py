from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama 
from pydantic import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Question
questionPromt = input("Enter your question: ")

# Connect to the Graph Database (Neo4j)
graph = Neo4jGraph(url="bolt://localhost:7687",
                   username="neo4j", password="neo4jneo4j")

# Connect to the LLM and set the context window
llm = OllamaFunctions(
    model="llama3.1", temperature=0, format="json", num_ctx=2048)

# Add vectors to the documents for vector search
vector_index  = Neo4jVector.from_existing_graph(
    OllamaEmbeddings(model="mxbai-embed-large"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jneo4j"
)

vector_retriever  = vector_index.as_retriever()

# Retrieve wthe Entities within the promt
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

#What about relations can they be extracted as well?
entity_chain = llm.with_structured_output(Entities)

# Test the entity extraction chain
# try:
#     entities = entity_chain.invoke(questionPromt) 
#     print(entities) # result names=['XXXX']
# except Exception as e:
#     print("Graph retriever failed: ", e)

# Fulltext index query
def graph_retriever(question):
    result = ""
    try:
        entities = entity_chain.invoke(questionPromt) 
        print("Entities: ", entities) # result names=['XXXX']
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
                RETURN output LIMIT 5000
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])
    except Exception as e:
        print("Graph retriever failed: ", e)
    return result

#Test the graph retriever   
print(graph_retriever(questionPromt))

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

def full_retriever(question: str): #Hybrid between graph and vector data of the documents
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""Relationships:\n{graph_data}\nMore information:\n {chr(10) + "".join(vector_data)}"""
    print(f"############################### \n {final_data} \n###############################")
    return final_data

template = """Answer the question based only on the following context: {context}\n
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
print(chain.invoke(input=questionPromt))
