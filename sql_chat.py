from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from get_model import get_bedrock_model

# Database URI
db_uri_postgres = "postgresql+psycopg2://postgres:password@localhost:5432/hiredb"
db = SQLDatabase.from_uri(db_uri_postgres)


# Get Schema
def get_schema(_):
    return db.get_table_info()


# Initialize LLM
llm = get_bedrock_model()

# Prompt Templates
templates = {
    'low': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where any of the provided skills {skills} are in the job's skills.

    SQL Query (without any explanations or formatting make a single line query):
    ''',

    'medium': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} match with the job's skills.

    SQL Query (without any explanations or formatting make a single line query):
    ''',

    'intermediate': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where any of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column.

    SQL Query (without any explanations or formatting make a single line query):
    ''',

    'high': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column.

    SQL Query (without any explanations or formatting make a single line query):
    '''
}


# Function to generate SQL query based on the provided template and query text
def generate_sql(template_name, skills, experience=None):
    prompt = ChatPromptTemplate.from_template(templates[template_name])
    sql_chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )

    if template_name in ['intermediate', 'high']:
        generated_sql = sql_chain.invoke({"skills": skills, "experience": experience})
    else:
        generated_sql = sql_chain.invoke({"skills": skills})

    if 'SQL Query:' in generated_sql:
        generated_sql = generated_sql.split('SQL Query:')[1].strip()

    return generated_sql


def run_query(query):
    return db.run(query)


# Functions for different match types

def return_low_matched_jobs(skills):
    sql_query = generate_sql('low', skills)
    print(sql_query)
    return run_query(sql_query)


def return_medium_matched_jobs(skills):
    sql_query = generate_sql('medium', skills)
    print(sql_query)
    return run_query(sql_query)


def return_intermediate_matched_jobs(skills, experience):
    sql_query = generate_sql('intermediate', skills, experience)
    print(sql_query)
    return run_query(sql_query)


def return_high_matched_jobs(skills, experience):
    sql_query = generate_sql('high', skills, experience)
    print(sql_query)
    return run_query(sql_query)


# Example usage
skills = "Python, Java, SQL"
experience = "5 years"

low_matched_jobs = return_low_matched_jobs(skills)
print(low_matched_jobs)

# medium_matched_jobs = return_medium_matched_jobs(skills)
# print("Medium Matched Jobs:", medium_matched_jobs)

intermediate_matched_jobs = return_intermediate_matched_jobs(skills, experience)
print(intermediate_matched_jobs)

high_matched_jobs = return_high_matched_jobs(skills, experience)
print(high_matched_jobs)