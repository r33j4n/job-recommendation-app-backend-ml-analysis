from langchain.prompts import ChatPromptTemplate
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
import json

CHROMA_DB_PATH = "database"



PROMPT_TEMPLATE = """
Fetch the necessary details from the following context the context is a Resume of a Job Seeker Therefore See the context in the sense of Resume :

{context}

Output as a JSON
---

{question}
"""

CHAT_PROMPT_TEMPLETE='''
As a Job Advisor Advice and provide Insights on my queries

{question}

'''


def query_ragcv():
    query_text = """

[Your Role]

You are an AI assistant specializing in extracting information from resumes/CVs. Your task is to analyze the provided CV Data which is in the context text and populate a structured JSON object based on a predefined schema.

[Schema]

The schema you need to adhere to is as follows:

```json
{    
    "skills": [],
    "experience": {
        "Years": [], 
        "Details": [
            {
                "CompanyName": "",
                "Role": "",
                "Duration": "" 
            }
        ]
    },
    "education": [
        {
            "Degree": "",
            "Institution": "",
            "GraduationYear": null 
        }
    ],

}
[Instructions]
Extract Information: Carefully read the CV text and identify the relevant details to fill in the schema fields.
Populate JSON: Fill in the JSON object with the extracted information. If a field is not mentioned in the CV, leave it as null.
Skills List: Create a list of skills from the "Technical Skills" section of the CV Skills need to be seperated by comma.
Experience Details: Extract company name, role, and duration (if available) from the "Working Experiences" section. If the duration is not explicitly stated, make a reasonable estimate based on the context.
Education Details: Extract degree, institution, and graduation year (if available) from the "Education" section.
Output: Return the completed JSON object as your final output.

        """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)

    # Extract the JSON part from the response
    start_index = response_text.find("{")
    end_index = response_text.rfind("}") + 1
    json_string = response_text[start_index:end_index]

    # Convert the JSON string into an actual JSON object
    json_response = json.loads(json_string)

    return json_response


def chat(query_text):

    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    # results = db.similarity_search_with_score(query_text, k=5)
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLETE)
    prompt = prompt_template.format(question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)

    # Extract the JSON part from the response


    return response_text

