from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from get_model import get_bedrock_model
from skills_experience_preprocess import parse_skills,extract_numeric_experience
from sqlalchemy import create_engine, text

# Database URI
db_uri_postgres = "postgresql+psycopg2://postgres:password@localhost:5432/hiredb"
db = SQLDatabase.from_uri(db_uri_postgres)
engine = create_engine("postgresql+psycopg2://postgres:password@localhost:5432/hiredb")


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

    Question: Given the SQL schema, return the job_ids from the job table where at least 3 of the provided skills {skills} Use the following format:
    Example Query: SELECT j.job_id FROM job j WHERE string_to_array(j.job_skills, ',') && ARRAY['Java', 'Python', 'C', 'React', 'Microsoft SQL server', 'Microsoft ASP.NET (C#)', 'Javascript', 'Software design', 'Code reviews', 'Testing', 'Video Streaming Technologies', 'C++', 'Unix', 'Linux'];
    Generate a very similar query with the given {skills} replacing the placeholder values. Note that string_to_array(js.skills, ',') part is Must needed because It need to convert to array (without any explanations or formatting make a cumpulsory single line query):
    ''',

    'medium': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} match with the job's skills.

    Postgres SQL Query (without any explanations or formatting make a single line query):
    ''',

    'intermediate': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: Given the SQL schema, return the job_ids from the job table where at least 3 of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column. Use the following format:
    Example Query: SELECT j.job_id FROM job j JOIN application a ON j.job_id = a.job_id JOIN jobseeker js ON a.job_seeker_id = js.job_seeker_id WHERE string_to_array(js.skills, ',') && ARRAY['Python', 'Java', 'C', 'JavaScript', 'Spring Boot', 'Flask', 'TensorFlow', 'Bootstrap', 'ReactJS', 'Scikit-learn', 'Pandas', 'NumPy', 'Git', 'GitHub', 'JIRA', 'Photoshop', 'Cloud Technologies', 'Virtual and Augmented Reality', 'Blockchain', 'Prompt Engineering', 'IoT', 'Sports Analytics'] AND j.job_experience LIKE '%5 year%';
    Generate a very similar query with the given {skills} and {experience} replacing the placeholder values. Note that string_to_array(js.skills, ',') part is Must needed because It need to convert to array (without any explanations or formatting make a cumpulsory single line query):
    ''',

    'high': ''' 
    Based on the table schema below, write a plain SQL query that would answer the user's question.
    {schema}

    Question: According to the SQL schema, I need to return the job_ids from the job table where at least 3 of the provided skills {skills} and the provided experience {experience} match with the job's skills and experience column.

    SQL Query (without any explanations or formatting make a single line query):
    '''
}


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
    print("Lower Matches Query is: ",sql_query)
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

#Testing
# skills = "Python, Java, SQL"
# experience = "2 Years"
# low_matched_jobs = return_low_matched_jobs(skills)
# print(low_matched_jobs)
# intermediate_matched_jobs = return_intermediate_matched_jobs(skills, experience)
# print(intermediate_matched_jobs)



def perform_matching(skills, experience):
    # Parse job seeker data
    js_skills = skills
    js_experience =experience

    # Fetch job data using SQLAlchemy
    with engine.connect() as connection:
        result = connection.execute(text("SELECT job_id, job_skills, job_experience FROM job"))
        jobs = result.fetchall()

    matched_job_ids = []
    for job in jobs:
        job_id = job[0]
        job_skills_str = job[1]
        job_experience_str = job[2]

        job_skills = parse_skills(job_skills_str)
        job_experience = extract_numeric_experience(job_experience_str)
        print(" Job Id: ",job_id," Job required Experience :",job_experience)
        matching_skills = js_skills & job_skills
        print(" Matching Skills :",matching_skills)
        num_matching_skills = len(matching_skills)
        experience_difference = js_experience - job_experience

        if ((num_matching_skills >= 2) and (experience_difference >=0)):
            print("Experience Difference : ",experience_difference)
            matched_job_ids.append((job_id,))

    print("High Matches",matched_job_ids)
    return matched_job_ids

def perform_low_matching(skills):
    js_skills=skills
    with engine.connect() as connection:
        result = connection.execute(text("SELECT job_id, job_skills, job_experience FROM job"))
        jobs = result.fetchall()
        matched_job_ids = []
        for job in jobs:
            job_id = job[0]
            job_skills_str = job[1]
            job_skills = parse_skills(job_skills_str)
            matching_skills = js_skills & job_skills
            num_matching_skills = len(matching_skills)

            if (num_matching_skills >= 1):
                matched_job_ids.append((job_id,))

        print("Low Matches", matched_job_ids)
        return matched_job_ids


