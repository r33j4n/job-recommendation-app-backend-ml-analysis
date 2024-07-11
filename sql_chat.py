import ast
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from get_model import get_bedrock_model

# Initialize database and prompt template
template = ''' 
Based on the table schema below, write a plain SQL query that would answer the user's question.
{schema}

Question: {question}

SQL Query (without any explanations or formatting):
'''

prompt = ChatPromptTemplate.from_template(template)

db_uri = "mysql+mysqlconnector://root:password@localhost:3306/hiredb"
db = SQLDatabase.from_uri(db_uri)

def get_schema(_):
    return db.get_table_info()

llm = get_bedrock_model()

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
)

def run_query(query):
    with db._engine.connect() as connection:
        result = connection.execute(query)
        return result.fetchall()

def generate_and_run_query(question):
    # Create a chain to generate the SQL query from the question
    generated_sql = sql_chain.invoke({"question": question})

    # Extract only the SQL query part if needed
    # This step depends on the output format of `llm`, adjust the logic accordingly
    if 'SQL Query:' in generated_sql:
        generated_sql = generated_sql.split('SQL Query:')[1].strip()

    print("Generated SQL Query:", generated_sql)

    # Run the extracted SQL query
    # result = run_query(generated_sql)
    # print(result)
    #
    # integers_list = [t[0] for t in result]
    #
    # # Convert integers_list to string
    # char_string = ''.join(map(str, integers_list))
    #
    # # Step 2: Use ast.literal_eval to safely evaluate the string as a Python expression
    # tuples_list = ast.literal_eval(char_string)
    #
    # # Step 3: Transform the list of tuples into a list of integers
    # final_list = [t[0] for t in tuples_list]

    return generated_sql
