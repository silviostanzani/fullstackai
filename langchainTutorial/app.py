import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

os.environ['API_KEY'] = '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'

model_id = 'tiiuae/falcon-7b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})
                            

template = """

You are an AI assistant that provides helpful answers to user queries.

{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])


falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)
                        
                        
print(falcon_chain.run("What are the colors in the Rainbow?"))

print(falcon_chain.run("What is python?"))

'''
@cl.langchain_factory(use_async=False)

def factory():

    prompt = PromptTemplate(template=template, input_variables=['question'])
    falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)

    return falcon_chain
'''

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()
