from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from langchain.llms import AzureOpenAI
import gradio as gr
import sys
import os
import openai
from typing import List

# import os
# import openai
# openai.api_type = "azure"
# openai.api_base = "https://hackmee1-we.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.ChatCompletion.create(
#   engine="HackWizardsGPT35",
#   messages = [],
#   temperature=0.7,
#   max_tokens=800,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None)


os.environ["OPENAI_API_KEY"] = '6d0dcfe671474d9d9dc030020b1d3203'

# llm = openai.ClassificationModel('gpt-3.5-turbo')


class NewAzureOpenAI(AzureOpenAI):
    stop: List[str] = None
    @property
    def _invocation_params(self):
        params = super()._invocation_params
        # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
        params.pop('logprobs', None)
        params.pop('best_of', None)
        params.pop('echo', None)
        #params['stop'] = self.stop
        return params

    
    
def construct_index(directory_path):
    
    #os.environ["OPENAI_API_TYPE"] = "azure"
    #os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    #os.environ["OPENAI_API_BASE"] = "https://hackmee1-we.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "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"
#     print(openai.ChatCompletion.create(
#       engine="HackWizardsGPT35",
#       messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"hii"},{"role":"assistant","content":"Hello! How can I assist you today?"}],
#       temperature=0.7,
#       max_tokens=800,
#       top_p=0.95,
#       frequency_penalty=0,
#       presence_penalty=0,
#       stop=None))
    
#     openai.api_type = "azure"
#     openai.api_base = "https://hackmee1-we.openai.azure.com/"
#     openai.api_version = "2023-03-15-preview"
#     openai.api_key = os.getenv("OPENAI_API_KEY")

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    
#     print(openai.api_key)

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm = openai.ClassificationModel('gpt-3.5-turbo')
    llm = OpenAI(
        model_name="text-davinci-003"
    )
#     print(llm("what is common noun? and be very specific and short in answer"))
    llm_predictor = LLMPredictor(llm=llm)

    print(llm_predictor)
#     llm_predictor = LLMPredictor(llm=AzureOpenAI(deployment_name="HackWizardsGPT35", model_name="gpt-35-turbo"))

   #llm_predictor = LLMPredictor(llm=llm)
    #service_context = ServiceContext.from_defaults(chunk_size_limit=2000,llm_predictor=llm_predictor)
    
    documents = SimpleDirectoryReader(directory_path).load_data()
#     print(documents)
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    #index = GPTSimpleVectorIndex.from_documents(documents,service_context=service_context)
    print(index)
    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    print('I am here1')
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    print('I am here2')
    response = index.query(input_text, response_mode="compact")
    print('I am here3')
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Meeshopedia limited to laap")

index = construct_index("data1")
iface.launch(share=True)