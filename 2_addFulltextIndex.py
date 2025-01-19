from neo4j import GraphDatabase

# Create Full Text embeddings
driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "neo4jneo4j"))

# Create the Embeddings
def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:!Document) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Function to execute the query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")


# Call the function to create the index
try:
    create_index()
except:
    pass

# Close the driver connection
driver.close()
