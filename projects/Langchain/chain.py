from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

import argparse
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="return a list of numbers")
args = parser.parse_args()     

llm = OpenAI()
#result = llm("write a very short poem")
#print (result)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

validation_prompt = PromptTemplate(
    template="Write a test in {language} for the following code:\n{code}",
    input_variables=["language", "code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",
)
validation_chain = LLMChain(
    llm=llm,
    prompt=validation_prompt,
    output_key="validation",
)

chain = SequentialChain(chains=[code_chain, validation_chain],
                              input_variables=["language", "task"],
    output_variables=["code", "validation"]
)

#result = code_chain({
#    "language": args.language,
#    "task": args.task
#})

result = chain({
    "language": args.language,
    "task": args.task
})

print(result["code"])
print(result["validation"])
#print(result["text"])