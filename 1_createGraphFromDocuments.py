from langchain_neo4j import Neo4jGraph
# from langchain_ollama import ChatOllama # ChatOllama doesn't work -> https://github.com/langchain-ai/langchainjs/issues/6051
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker # https://python.langchain.com/docs/how_to/semantic-chunker/

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain_core.documents import Document
import re

# Connect to the llm and set the context window 
extraction_llm = OllamaFunctions( 
    model="llama3.1", temperature=0, format="json", num_ctx=16834)

#Load embedding model for semantic chunker
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

# Connect to the graph database (Neo4j)
graph = Neo4jGraph(url="bolt://localhost:7687",
                   username="neo4j", password="neo4jneo4j")

# Delete all nodes in the graph 
graph.query("""MATCH (n) DETACH DELETE n""")

# 3 Ways of loading documents

documentname = "ISO27001.pdf"

# 1) load pdf into documents   
async def load_documents():
    documents = []

    loader = PyPDFLoader("./Documents/"+ documentname)
    async for page in loader.alazy_load():
        documents.append(Document(page_content=page.page_content))
    return documents
documents = asyncio.run(load_documents())

# 2) load from multiple Text files
# memeDescriptions = glob.glob("descriptions/*", recursive=True)
# for path in memeDescriptions:
#     loader = TextLoader(file_path=path)
# documents = loader.load()

# 3) load single Text file
# documents = TextLoader(file_path="./Documents/enslavement.txt").load()
# print(f"###################################### Documents ######################################\n {documents}")
# text = re.sub('[^A-Za-z0-9 ]+', '', documents[0].page_content)
# txtlen = len(text)
# documents = [Document(page_content=text)]

def getGraph(chunk_size, chunk_overlap):

    # Use a chunking strategy 
    # https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # chunks = text_splitter.split_documents(documents=documents)

    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunks = semantic_chunker.create_documents([d.page_content for d in documents])
    for semantic_chunk in chunks:
        if "Effect of Pre-training Tasks" in semantic_chunk.page_content:
            print(semantic_chunk.page_content)
            print(len(semantic_chunk.page_content))

    print(
        f"###################################### Chunks ######################################\n {chunks}")

    # # Define schema
    # allowed_nodes = ["Person", "NobelPrize", "Discovery", "Document"]
    # allowed_relationships = [
    #     ("Person", "IS_MARRIED_TO", "Person"),
    #     ("Person", "WORKED_WITH", "Person"),
    #     ("Person", "WON", "NobelPrize"),
    #     ("Person", "DISCOVERED", "Discovery"),
    #     ("Document", "MENTIONS", "*")
    # ]
    print(
        f"###################################### Model ######################################")

    llm_transformer = LLMGraphTransformer(llm=extraction_llm)
    #   allowed_nodes=allowed_nodes,
    #   allowed_relationships=allowed_relationships)
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)

    print(
        f"###################################### Graph ######################################\n {graph_documents}")

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

# Fibbonacci? have chunks sumarized by llm?
# while txtlen > 100:
#     print(f"###################################### {txtlen} ######################################")
#     getGraph(txtlen, round(txtlen * 0.618))
#     txtlen = round(txtlen * 0.618)

getGraph(0,0)

# graph.query("""MATCH (n) WHERE not( (n)-[]-() ) DELETE n""")

# Idea link all to the document
# graph.query("""MATCH (n)
# MATCH (newNode:Whole_Document {name: '_dummytext'})
# MERGE (n)-[:MENTIONS]->(newNode)""")