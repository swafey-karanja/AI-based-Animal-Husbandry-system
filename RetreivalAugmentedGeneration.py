from gradient_haystack.embedders.gradient_document_embedder import GradientDocumentEmbedder
from gradient_haystack.embedders.gradient_text_embedder import GradientTextEmbedder
from gradient_haystack.generator.base import GradientGenerator
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
import os
# import requests

os.environ['GRADIENT_ACCESS_TOKEN'] = "4RkXwcXCIhjSilcrkYNanvSI8h1WWrgt"
os.environ['GRADIENT_WORKSPACE_ID'] = "496b8f01-47f9-4f62-91c8-e634679ca2d3_workspace"

fine_tuned_Model_Id = "28643f93-bdd5-4602-b911-2e9fea183186_model_adapter"

document_store = InMemoryDocumentStore()
writer = DocumentWriter(document_store=document_store)

document_embedder = GradientDocumentEmbedder(
    access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
)

# URL of the online repository where the Raw_Text_Data.txt file is located
# url = "https://raw.githubusercontent.com/swafey-karanja/Model-training/main/Raw_Text_Data.txt"

# # Send a GET request to download the file
# response = requests.get(url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Read the contents of the downloaded file
#     text_data = response.text
# else:
#     # If the request was not successful, print an error message
#     print("Failed to download the file from the URL:", url)

with open("Raw_Text_Data.txt", encoding="utf-8") as file:
    text_data = file.read()

docs = [
    Document(content=text_data)
]

print(len(text_data))

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(
    instance=document_embedder, name="document_embedder")
indexing_pipeline.add_component(instance=writer, name="writer")
indexing_pipeline.connect("document_embedder", "writer")
indexing_pipeline.run({"document_embedder": {"documents": docs}})

text_embedder = GradientTextEmbedder(
    access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
)

generator = GradientGenerator(
    access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    model_adapter_id=fine_tuned_Model_Id,
    max_generated_token_count=350,
)

prompt = """You are helpful assistant meant to answer questions relating to
animal husbandry. Answer the query, based on the
content in the documents. if you dont know the answer respond by saying you
are unable to assist with that at the moment.
{{documents}}
Query: {{query}}
\nAnswer:
"""

retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt)

rag_pipeline = Pipeline()
rag_pipeline.add_component(instance=text_embedder, name="text_embedder")
rag_pipeline.add_component(instance=retriever, name="retriever")
rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
rag_pipeline.add_component(instance=generator, name="generator")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
rag_pipeline.connect("generator.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")


def LLM_Run(question):
    result = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
            "answer_builder": {"query": question}
        }
    )
    return result["answer_builder"]["answers"][0].data


Query = "Do bulls show signs of Trichomoniasis?"
print(LLM_Run(Query))
