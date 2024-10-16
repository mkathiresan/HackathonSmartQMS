import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
CHROMA_URL_PATH = "urlchroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}


"""

PROMPT_TEMPLATE_URL = """

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}


"""

PROMPT_COMPARE = """

Compare the answers from below context and find any differences in it.

Answer 1: {answer_pdf}

Answer 2: {answer_url}

"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def remove_duplicate(sources):
    truncated = [x.replace('data\\', '') for x in sources]
    truncated1 = [x.split('.pdf')[0] for x in truncated]
    rem_dup = list(set(truncated1))
    return rem_dup


def remove_dup_url(sources):
    truncated1 = [x.split(':None')[0] for x in sources]
    rem_dup = list(set(truncated1))
    return rem_dup


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    db_url = Chroma(persist_directory=CHROMA_URL_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Search in URL DB
    results_url = db_url.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    url_context_text = "\n\n---\n\n".join([doc_url.page_content for doc_url, _score_url in results_url])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    unique_source = remove_duplicate(sources)
    formatted_response = f"Response: {response_text}\nSources: {unique_source}"
    print(formatted_response)

    # compare the results with URL
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_URL)
    prompt = prompt_template.format(question=query_text, context=url_context_text)
    # print(prompt)

    model = Ollama(model="llama3.2")
    response_text_url = model.invoke(prompt)

    sources_url = [doc.metadata.get("id", None) for doc, _score in results_url]
    sources_url_dup = remove_dup_url(sources_url)
    formatted_response_url = f"Response: {response_text_url}\nSources: {sources_url_dup}"
    print(formatted_response_url)

    # compare the results with URL
    prompt_template_compare = ChatPromptTemplate.from_template(PROMPT_COMPARE)
    prompt = prompt_template_compare.format(answer_pdf=response_text, answer_url=response_text_url)
    # print(prompt)

    model = Ollama(model="llama3.2")
    compare_text = model.invoke(prompt)
    print(compare_text)
    return response_text, unique_source + sources_url_dup, compare_text, response_text_url


if __name__ == "__main__":
    main()
