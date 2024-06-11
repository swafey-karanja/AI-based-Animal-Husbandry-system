from langchain.chains import LLMChain
# Import the LLMChain class for building LLM-based workflows
from langchain.llms import GradientLLM
# Import the GradientLLM class for interacting with Gradient AI's API
from langchain.prompts import PromptTemplate
# Import the PromptTemplate class for defining how to prompt the LLM
# import gradientai
# import os # Import the os module for potential file system interactions

Fine_Tune__adapter_ID = "28643f93-bdd5-4602-b911-2e9fea183186_model_adapter"
# Fine_Tune__adapter_ID = Fine_Tune__adapter.id
#  creating a GradientLLM object
llm = GradientLLM(
    model=Fine_Tune__adapter_ID,
    model_kwargs=dict(
        max_generated_token_count=128,
        # Adjust how your model generates completions
        temperature=0.7,
        # randomness
        top_k=50  # Restricts the model to pick from k most likely words,
    ),
)

template = """### Instruction: {Instruction} \n\n### Response:"""

prompt = PromptTemplate(template=template, input_variables=["Instruction"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
