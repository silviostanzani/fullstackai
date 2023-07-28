# langchain tutorial

## install packages
```
pip install huggingface_hub 
pip install transformers
pip install langchain
pip install chainlit
```

## hello world (chainlit)
```
chainlit hello
```

## simple prompt

### Steps to create simple prompt

#### imports
```
import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
```

#### choose llm model
```
os.environ['API_KEY'] = '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'

model_id = 'tiiuae/falcon-7b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})
```

#### define prompt format
```
template = """

AI assistant based on falcon-7b

{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])

falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)
```

#### configure execution
```
@cl.langchain_factory(use_async=False)

def factory():

    prompt = PromptTemplate(template=template, input_variables=['question'])
    falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)

    return falcon_chain
```

#### start server
```
chainlit run app.py -w
```


